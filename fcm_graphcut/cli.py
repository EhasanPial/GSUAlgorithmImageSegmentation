"""Command-line entry point.

Example:
    python -m fcm_graphcut.cli --image input.png --out out_dir --gt gt.png
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from .segment import segment, dice_score, iou_score
from .viz import make_panel, overlay_mask


def load_gray(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float64)


def load_mask(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.array(img)
    return (arr > 127).astype(np.uint8)


def save_mask(mask: np.ndarray, path: str) -> None:
    Image.fromarray((mask * 255).astype(np.uint8)).save(path)


def save_overlay(image: np.ndarray, mask: np.ndarray, path: str) -> None:
    ov = (overlay_mask(image, mask) * 255).astype(np.uint8)
    Image.fromarray(ov).save(path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fcm_graphcut",
        description="FCM + Graph Cut binary image segmentation.",
    )
    p.add_argument("--image", required=True, help="Path to input grayscale image.")
    p.add_argument("--out", default="output", help="Output directory (created if missing).")
    p.add_argument("--gt", default=None, help="Optional ground-truth binary mask (for Dice/IoU).")

    p.add_argument("--fg-mask", default=None,
                   help="Optional binary mask: white pixels are LOCKED as foreground.")
    p.add_argument("--bg-mask", default=None,
                   help="Optional binary mask: white pixels are LOCKED as background.")

    p.add_argument("--clusters", type=int, default=2, help="Number of FCM clusters.")
    p.add_argument("--m", type=float, default=2.0, help="FCM fuzzifier (m > 1).")
    p.add_argument("--sigma", type=float, default=30.0,
                   help="Gaussian stddev for n-link weights (intensity scale).")
    p.add_argument("--lambda-data", type=float, default=1.0, help="Data term weight.")
    p.add_argument("--lambda-smooth", type=float, default=20.0,
                   help="Smoothness term weight (larger = smoother).")
    p.add_argument("--resize", type=int, default=None,
                   help="Optional: resize longest side to this many pixels (speeds up).")
    p.add_argument("--solver", choices=["ek", "dinic", "bk"], default="ek",
                   help="Max-flow solver: 'ek' = Edmonds-Karp, 'dinic' = Dinic's, 'bk' = Boykov-Kolmogorov.")
    p.add_argument("--invert-gt", action="store_true",
                   help="Invert GT mask (if foreground is 0 instead of 255).")
    p.add_argument("--seed", type=int, default=42)
    return p


def maybe_resize(arr: np.ndarray, long_side: int) -> np.ndarray:
    h, w = arr.shape
    m = max(h, w)
    if m <= long_side:
        return arr
    scale = long_side / m
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    img = Image.fromarray(arr.astype(np.uint8) if arr.dtype != np.uint8 else arr)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    return np.array(img, dtype=arr.dtype)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] image: {args.image}")
    image = load_gray(args.image)
    if args.resize is not None:
        image = maybe_resize(image, args.resize)
    print(f"       shape: {image.shape}, range: [{image.min():.1f}, {image.max():.1f}]")

    gt = None
    if args.gt is not None:
        print(f"[load] gt:    {args.gt}")
        gt = load_mask(args.gt)
        if args.resize is not None:
            gt = maybe_resize(gt, args.resize)
            gt = (gt > 0).astype(np.uint8)
        if args.invert_gt:
            gt = 1 - gt
        if gt.shape != image.shape:
            raise ValueError(f"GT shape {gt.shape} != image shape {image.shape}")

    fg_seeds = bg_seeds = None
    if args.fg_mask is not None:
        print(f"[load] fg seeds: {args.fg_mask}")
        fg_seeds = load_mask(args.fg_mask)
        if args.resize is not None:
            fg_seeds = maybe_resize(fg_seeds, args.resize)
            fg_seeds = (fg_seeds > 0).astype(np.uint8)
        if fg_seeds.shape != image.shape:
            raise ValueError(f"FG mask shape {fg_seeds.shape} != image {image.shape}")
        print(f"       fg seed pixels: {int(fg_seeds.sum())}")
    if args.bg_mask is not None:
        print(f"[load] bg seeds: {args.bg_mask}")
        bg_seeds = load_mask(args.bg_mask)
        if args.resize is not None:
            bg_seeds = maybe_resize(bg_seeds, args.resize)
            bg_seeds = (bg_seeds > 0).astype(np.uint8)
        if bg_seeds.shape != image.shape:
            raise ValueError(f"BG mask shape {bg_seeds.shape} != image {image.shape}")
        print(f"       bg seed pixels: {int(bg_seeds.sum())}")
    if fg_seeds is not None and bg_seeds is not None:
        overlap = int(np.logical_and(fg_seeds, bg_seeds).sum())
        if overlap > 0:
            print(f"[warn] {overlap} pixels are marked as BOTH FG and BG; "
                  f"FG will take precedence in graph.")

    print(f"[segment] solver={args.solver} clusters={args.clusters} m={args.m} sigma={args.sigma} "
          f"lambda_data={args.lambda_data} lambda_smooth={args.lambda_smooth}")
    t0 = time.time()
    result = segment(
        image,
        n_clusters=args.clusters,
        m=args.m,
        sigma=args.sigma,
        lambda_data=args.lambda_data,
        lambda_smooth=args.lambda_smooth,
        seed=args.seed,
        fg_seeds=fg_seeds,
        bg_seeds=bg_seeds,
        solver=args.solver,
    )
    dt = time.time() - t0
    print(f"[done] max-flow value = {result.max_flow_value:.4f}   elapsed = {dt:.2f}s")
    print(f"       cluster centers = {np.round(result.centers, 2)}")
    print(f"       foreground pixels = {int(result.mask.sum())} / {result.mask.size}")

    # save outputs
    mask_path = out_dir / "mask.png"
    overlay_path = out_dir / "overlay.png"
    panel_path = out_dir / "panel.png"
    save_mask(result.mask, str(mask_path))
    save_overlay(image, result.mask, str(overlay_path))

    dice = iou = None
    if gt is not None:
        dice = dice_score(result.mask, gt)
        iou = iou_score(result.mask, gt)
        print(f"[metrics] Dice = {dice:.4f}   IoU = {iou:.4f}")

        # Also try the inverse labeling in case object/background are swapped
        dice_inv = dice_score(1 - result.mask, gt)
        if dice_inv > dice:
            print(f"[note] inverse labeling gives Dice = {dice_inv:.4f} "
                  f"(> {dice:.4f}); flipping mask.")
            result.mask[:] = 1 - result.mask
            dice = dice_inv
            iou = iou_score(result.mask, gt)
            save_mask(result.mask, str(mask_path))
            save_overlay(image, result.mask, str(overlay_path))
            print(f"[metrics] Dice = {dice:.4f}   IoU = {iou:.4f}  (after flip)")

    fig = make_panel(image, result.mask, result.U, result.obj_idx,
                     gt=gt, dice=dice, iou=iou)
    fig.savefig(panel_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"[write] {mask_path}")
    print(f"[write] {overlay_path}")
    print(f"[write] {panel_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())