import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def psnr_from_mse(mse: float) -> float:
    return -10.0 * math.log10(max(mse, 1e-12))


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def foreground_mask_from_gt(gt: np.ndarray) -> np.ndarray:
    """Use the black transparent-background render as a foreground mask."""
    return np.any(gt > 1.0 / 255.0, axis=-1)


def main():
    parser = argparse.ArgumentParser(description="Evaluate official 3DGS rendered images.")
    parser.add_argument("--render_dir", required=True, help="Directory containing renders/ and gt/.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--save_examples", type=int, default=8)
    parser.add_argument(
        "--foreground_only",
        action="store_true",
        help="Ignore black background pixels in GT when computing metrics.",
    )
    args = parser.parse_args()

    render_dir = Path(args.render_dir)
    renders = sorted((render_dir / "renders").glob("*.png"))
    gts = sorted((render_dir / "gt").glob("*.png"))
    if len(renders) != len(gts):
        raise ValueError(f"renders={len(renders)} gt={len(gts)}")

    output_dir = Path(args.output_dir)
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    l1_values = []
    mse_values = []
    for idx, (render_path, gt_path) in enumerate(zip(renders, gts)):
        rendered = read_rgb(render_path)
        gt = read_rgb(gt_path)
        diff = rendered - gt
        if args.foreground_only:
            mask = foreground_mask_from_gt(gt)
            if not np.any(mask):
                raise ValueError(f"empty foreground mask for {gt_path}")
            diff_eval = diff[mask]
        else:
            diff_eval = diff
        l1_values.append(float(np.abs(diff_eval).mean()))
        mse_values.append(float((diff_eval * diff_eval).mean()))

        if idx < args.save_examples:
            vis = np.concatenate([gt, rendered], axis=1)
            vis = (vis * 255).clip(0, 255).astype(np.uint8)
            h, w = gt.shape[:2]
            cv2.putText(vis, "GT", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(vis, "Official 3DGS", (w + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.imwrite(str(examples_dir / f"view_{idx:03d}.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    metrics = {
        "render_dir": str(render_dir),
        "foreground_only": bool(args.foreground_only),
        "num_images": len(renders),
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
