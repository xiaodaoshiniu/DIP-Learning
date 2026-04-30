from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass
class BADataset:
    xy: torch.Tensor
    visibility: torch.Tensor
    colors: np.ndarray
    image_size: int
    view_names: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch Bundle Adjustment for DIP Assignment 03")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing points2d.npz and colors.")
    parser.add_argument("--output-dir", type=str, default="results/task1_ba", help="Output directory.")
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu.")
    parser.add_argument("--iters", type=int, default=3000, help="Number of Adam iterations.")
    parser.add_argument("--batch-size", type=int, default=262144, help="Visible 2D observations per iteration.")
    parser.add_argument("--lr-points", type=float, default=2e-2, help="Learning rate for 3D points.")
    parser.add_argument("--lr-camera", type=float, default=3e-3, help="Learning rate for camera extrinsics.")
    parser.add_argument("--lr-focal", type=float, default=1e-2, help="Learning rate for shared focal length.")
    parser.add_argument("--init-focal", type=float, default=900.0, help="Initial shared focal length.")
    parser.add_argument("--init-depth", type=float, default=2.5, help="Initial object depth relative to cameras.")
    parser.add_argument("--yaw-range-deg", type=float, default=70.0, help="Initial yaw range for all views.")
    parser.add_argument("--image-size", type=int, default=1024, help="Image size in pixels.")
    parser.add_argument("--max-views", type=int, default=0, help="Debug option: keep only first K views.")
    parser.add_argument("--max-points", type=int, default=0, help="Debug option: keep only first K points.")
    parser.add_argument("--log-every", type=int, default=50, help="Print/evaluate every K iterations.")
    parser.add_argument("--eval-chunk", type=int, default=524288, help="Chunk size for full-metric evaluation.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--robust-eps", type=float, default=1e-3, help="Charbonnier epsilon in normalized pixels.")
    parser.add_argument("--rotation-prior", type=float, default=1e-4, help="Weak prior to keep rotations near initialization.")
    parser.add_argument("--translation-prior", type=float, default=1e-4, help="Weak prior to keep translations near initialization.")
    parser.add_argument("--focal-prior", type=float, default=1e-4, help="Weak prior to keep focal length in a sane range.")
    parser.add_argument("--center-prior", type=float, default=1e-3, help="Weak prior to remove point-cloud gauge drift.")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_dataset(data_dir: Path, image_size: int, max_views: int, max_points: int, device: torch.device) -> BADataset:
    points2d = np.load(data_dir / "points2d.npz")
    view_names = sorted(points2d.files)
    if max_views > 0:
        view_names = view_names[:max_views]

    arrays = [points2d[name] for name in view_names]
    if max_points > 0:
        arrays = [arr[:max_points] for arr in arrays]

    stacked = np.stack(arrays, axis=0).astype(np.float32)
    colors = np.load(data_dir / "points3d_colors.npy").astype(np.float32)
    if max_points > 0:
        colors = colors[:max_points]
    if colors.max() > 1.0:
        colors = colors / 255.0

    return BADataset(
        xy=torch.from_numpy(stacked[:, :, :2]).to(device),
        visibility=torch.from_numpy(stacked[:, :, 2] > 0.5).to(device),
        colors=colors,
        image_size=image_size,
        view_names=view_names,
    )


def axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError(f"Unknown rotation axis: {axis}")
    return torch.stack(flat, dim=-1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler: torch.Tensor, convention: str = "XYZ") -> torch.Tensor:
    matrices = [axis_angle_rotation(axis, angle) for axis, angle in zip(convention, torch.unbind(euler, -1))]
    result = matrices[0]
    for matrix in matrices[1:]:
        result = torch.matmul(result, matrix)
    return result


def initialize_parameters(dataset: BADataset, args: argparse.Namespace, device: torch.device):
    n_views, n_points = dataset.xy.shape[:2]
    cx = cy = args.image_size / 2.0
    f0 = float(args.init_focal)
    depth = float(args.init_depth)

    center_view = n_views // 2
    center_xy = dataset.xy[center_view].detach().cpu().numpy()
    visible = dataset.visibility[center_view].detach().cpu().numpy()
    all_xy = dataset.xy.detach().cpu().numpy()
    all_vis = dataset.visibility.detach().cpu().numpy()

    # Initialize every 3D point from its center view projection if visible;
    # otherwise use the mean visible 2D position across all views.
    fallback_xy = np.zeros((n_points, 2), dtype=np.float32)
    for point_idx in range(n_points):
        point_vis = all_vis[:, point_idx]
        if np.any(point_vis):
            fallback_xy[point_idx] = all_xy[point_vis, point_idx, :2].mean(axis=0)
        else:
            fallback_xy[point_idx] = np.array([cx, cy], dtype=np.float32)
    init_xy = np.where(visible[:, None], center_xy, fallback_xy)

    x = (init_xy[:, 0] - cx) * depth / f0
    y = (cy - init_xy[:, 1]) * depth / f0
    z = np.random.default_rng(args.seed).normal(loc=0.0, scale=0.02, size=n_points)
    points = torch.nn.Parameter(torch.tensor(np.stack([x, y, z], axis=1), dtype=torch.float32, device=device))

    yaw = np.linspace(-math.radians(args.yaw_range_deg), math.radians(args.yaw_range_deg), n_views, dtype=np.float32)
    euler_init = np.zeros((n_views, 3), dtype=np.float32)
    euler_init[:, 1] = yaw
    translations_init = np.zeros((n_views, 3), dtype=np.float32)
    translations_init[:, 2] = -depth

    euler = torch.nn.Parameter(torch.tensor(euler_init, dtype=torch.float32, device=device))
    translations = torch.nn.Parameter(torch.tensor(translations_init, dtype=torch.float32, device=device))
    log_focal = torch.nn.Parameter(torch.tensor(math.log(f0), dtype=torch.float32, device=device))

    priors = {
        "euler": torch.tensor(euler_init, dtype=torch.float32, device=device),
        "translations": torch.tensor(translations_init, dtype=torch.float32, device=device),
        "log_focal": torch.tensor(math.log(f0), dtype=torch.float32, device=device),
    }
    return points, euler, translations, log_focal, priors


def project_points(
    points3d: torch.Tensor,
    euler: torch.Tensor,
    translations: torch.Tensor,
    focal: torch.Tensor,
    camera_indices: torch.Tensor,
    point_indices: torch.Tensor,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotations = euler_angles_to_matrix(euler[camera_indices], "XYZ")
    selected_points = points3d[point_indices]
    camera_points = torch.bmm(rotations, selected_points.unsqueeze(-1)).squeeze(-1) + translations[camera_indices]

    x, y, z = camera_points[:, 0], camera_points[:, 1], camera_points[:, 2]
    z_safe = torch.where(z.abs() < 1e-4, torch.full_like(z, -1e-4), z)
    cx = cy = image_size / 2.0
    u = -focal * x / z_safe + cx
    v = focal * y / z_safe + cy
    return torch.stack([u, v], dim=-1), z


def make_observation_index(dataset: BADataset) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cam_idx, point_idx = torch.nonzero(dataset.visibility, as_tuple=True)
    obs_xy = dataset.xy[cam_idx, point_idx]
    return cam_idx, point_idx, obs_xy


def robust_reprojection_loss(pred_xy: torch.Tensor, obs_xy: torch.Tensor, image_size: int, eps: float) -> torch.Tensor:
    residual = (pred_xy - obs_xy) / float(image_size)
    return torch.sqrt((residual * residual).sum(dim=-1) + eps * eps).mean() * float(image_size)


@torch.no_grad()
def evaluate_rmse(
    points3d: torch.Tensor,
    euler: torch.Tensor,
    translations: torch.Tensor,
    focal: torch.Tensor,
    cam_idx: torch.Tensor,
    point_idx: torch.Tensor,
    obs_xy: torch.Tensor,
    image_size: int,
    chunk_size: int,
) -> dict[str, float]:
    sq_sum = 0.0
    l2_sum = 0.0
    count = 0
    for start in range(0, obs_xy.shape[0], chunk_size):
        end = min(start + chunk_size, obs_xy.shape[0])
        pred, _ = project_points(points3d, euler, translations, focal, cam_idx[start:end], point_idx[start:end], image_size)
        diff = pred - obs_xy[start:end]
        sq_sum += float((diff * diff).sum().item())
        l2_sum += float(torch.linalg.norm(diff, dim=-1).sum().item())
        count += end - start
    return {
        "rmse_px": math.sqrt(sq_sum / max(1, 2 * count)),
        "mean_l2_px": l2_sum / max(1, count),
    }


def write_obj(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Reconstructed point cloud from PyTorch Bundle Adjustment\n")
        for point, color in zip(points, colors):
            r, g, b = np.clip(color, 0.0, 1.0)
            handle.write(f"v {point[0]:.7f} {point[1]:.7f} {point[2]:.7f} {r:.6f} {g:.6f} {b:.6f}\n")


def save_loss_curve(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = [row["iter"] for row in history]
    ys = [row["train_loss_px"] for row in history]
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Charbonnier reprojection loss (px)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_point_cloud_preview(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stride = max(1, len(points) // 6000)
    pts = points[::stride]
    cols = np.clip(colors[::stride], 0.0, 1.0)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c=cols, s=1.0)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.view_init(elev=12, azim=-80)
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_training_log(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    dataset = load_dataset(Path(args.data_dir), args.image_size, args.max_views, args.max_points, device)
    cam_idx, point_idx, obs_xy = make_observation_index(dataset)
    points3d, euler, translations, log_focal, priors = initialize_parameters(dataset, args, device)

    optimizer = torch.optim.Adam(
        [
            {"params": [points3d], "lr": args.lr_points},
            {"params": [euler, translations], "lr": args.lr_camera},
            {"params": [log_focal], "lr": args.lr_focal},
        ]
    )

    n_obs = obs_xy.shape[0]
    history: list[dict[str, float]] = []
    print(f"Loaded {len(dataset.view_names)} views, {dataset.xy.shape[1]} points, {n_obs} visible observations.")
    print(f"Running on {device}; output directory: {output_dir}")

    for iteration in range(1, args.iters + 1):
        if args.batch_size > 0 and args.batch_size < n_obs:
            sample = torch.randint(0, n_obs, (args.batch_size,), device=device)
        else:
            sample = torch.arange(n_obs, device=device)

        focal = torch.exp(log_focal)
        pred_xy, z = project_points(
            points3d,
            euler,
            translations,
            focal,
            cam_idx[sample],
            point_idx[sample],
            args.image_size,
        )
        train_loss = robust_reprojection_loss(pred_xy, obs_xy[sample], args.image_size, args.robust_eps)
        depth_penalty = torch.relu(z + 0.05).square().mean()
        prior_loss = (
            args.rotation_prior * (euler - priors["euler"]).square().mean()
            + args.translation_prior * (translations - priors["translations"]).square().mean()
            + args.focal_prior * (log_focal - priors["log_focal"]).square()
            + args.center_prior * points3d.mean(dim=0).square().sum()
            + 10.0 * depth_penalty
        )
        loss = train_loss + prior_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iteration == 1 or iteration % args.log_every == 0 or iteration == args.iters:
            metrics = evaluate_rmse(
                points3d,
                euler,
                translations,
                torch.exp(log_focal),
                cam_idx,
                point_idx,
                obs_xy,
                args.image_size,
                args.eval_chunk,
            )
            row = {
                "iter": float(iteration),
                "train_loss_px": float(train_loss.item()),
                "rmse_px": metrics["rmse_px"],
                "mean_l2_px": metrics["mean_l2_px"],
                "focal": float(torch.exp(log_focal).item()),
            }
            history.append(row)
            print(
                f"[{iteration:05d}/{args.iters}] "
                f"loss={row['train_loss_px']:.4f}px rmse={row['rmse_px']:.4f}px "
                f"mean_l2={row['mean_l2_px']:.4f}px f={row['focal']:.2f}"
            )

    final_points = points3d.detach().cpu().numpy()
    final_euler = euler.detach().cpu().numpy()
    final_translations = translations.detach().cpu().numpy()
    final_focal = float(torch.exp(log_focal).detach().cpu().item())

    write_obj(output_dir / "reconstruction.obj", final_points, dataset.colors)
    save_loss_curve(output_dir / "loss_curve.png", history)
    save_point_cloud_preview(output_dir / "point_cloud_preview.png", final_points, dataset.colors)
    save_training_log(output_dir / "training_log.csv", history)
    np.savez(
        output_dir / "camera_params.npz",
        euler=final_euler,
        translations=final_translations,
        focal=np.array([final_focal], dtype=np.float32),
        view_names=np.array(dataset.view_names),
    )

    final_metrics = history[-1] if history else {}
    summary = {
        "num_views": len(dataset.view_names),
        "num_points": int(dataset.xy.shape[1]),
        "num_visible_observations": int(n_obs),
        "image_size": args.image_size,
        "device": str(device),
        "iters": args.iters,
        "batch_size": args.batch_size,
        "final_focal": final_focal,
        "final_metrics": final_metrics,
        "outputs": {
            "obj": "reconstruction.obj",
            "loss_curve": "loss_curve.png",
            "preview": "point_cloud_preview.png",
            "camera_params": "camera_params.npz",
            "training_log": "training_log.csv",
        },
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Done.")
    print(f"OBJ: {output_dir / 'reconstruction.obj'}")
    print(f"Loss curve: {output_dir / 'loss_curve.png'}")
    print(f"Preview: {output_dir / 'point_cloud_preview.png'}")


if __name__ == "__main__":
    main()
