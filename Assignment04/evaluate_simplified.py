import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from data_utils import ColmapDataset
from gaussian_model import GaussianModel
from gaussian_renderer import GaussianRenderer


def psnr_from_mse(mse: float) -> float:
    return -10.0 * math.log10(max(mse, 1e-12))


def foreground_mask_from_gt(gt: torch.Tensor) -> torch.Tensor:
    """Use non-black GT pixels as the foreground mask for RGBA synthetic data."""
    return (gt > 1.0 / 255.0).any(dim=-1)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the simplified PyTorch 3DGS checkpoint.")
    parser.add_argument("--colmap_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_examples", type=int, default=8)
    parser.add_argument(
        "--foreground_only",
        action="store_true",
        help="Ignore black background pixels in GT when computing metrics.",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(exist_ok=True)

    dataset = ColmapDataset(args.colmap_dir)
    sample = dataset[0]["image"]
    H, W = sample.shape[:2]

    model = GaussianModel(dataset.points3D_xyz, dataset.points3D_rgb).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    renderer = GaussianRenderer(H, W).to(device)
    l1_values = []
    mse_values = []

    with torch.no_grad():
        gaussian_params = model()
        for idx in tqdm(range(len(dataset)), desc="Evaluating"):
            item = dataset[idx]
            gt = item["image"].to(device)
            rendered = renderer(
                means3D=gaussian_params["positions"],
                covs3d=gaussian_params["covariance"],
                colors=gaussian_params["colors"],
                opacities=gaussian_params["opacities"],
                K=item["K"].to(device),
                R=item["R"].to(device),
                t=item["t"].to(device).reshape(3),
            ).clamp(0.0, 1.0)

            diff = rendered - gt
            if args.foreground_only:
                mask = foreground_mask_from_gt(gt)
                if not bool(mask.any()):
                    raise ValueError(f"empty foreground mask for view {idx}")
                diff_eval = diff[mask]
            else:
                diff_eval = diff
            l1_values.append(diff_eval.abs().mean().item())
            mse_values.append((diff_eval * diff_eval).mean().item())

            if idx < args.save_examples:
                gt_np = (gt.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                rd_np = (rendered.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                vis = np.concatenate([gt_np, rd_np], axis=1)
                cv2.putText(vis, "GT", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(vis, "Simplified 3DGS", (W + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.imwrite(str(examples_dir / f"view_{idx:03d}.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    metrics = {
        "checkpoint": args.checkpoint,
        "foreground_only": bool(args.foreground_only),
        "num_images": len(dataset),
        "num_gaussians": int(model.n_points),
        "mean_l1": float(np.mean(l1_values)),
        "mean_mse": float(np.mean(mse_values)),
        "mean_psnr": psnr_from_mse(float(np.mean(mse_values))),
        "per_view_psnr": [psnr_from_mse(v) for v in mse_values],
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_view_psnr"}, indent=2))


if __name__ == "__main__":
    main()
