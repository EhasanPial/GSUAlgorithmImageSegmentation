"""Benchmark Edmonds-Karp vs Boykov-Kolmogorov on the same image(s).

Both solvers compute the exact min-cut, so segmentation quality (Dice/IoU)
is identical. The interesting comparison is RUNTIME and number of pixels
where the two masks disagree (should be 0).

Usage:
    python -m fcm_graphcut.benchmark --image samples/circle.png --gt samples/circle_gt.png
    python -m fcm_graphcut.benchmark --image samples/circle.png --gt samples/circle_gt.png --resize 64 96 128
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .segment import segment, dice_score, iou_score


def load_gray(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float64)


def load_mask(path: str) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return (arr > 127).astype(np.uint8)


def resize(arr: np.ndarray, long_side: int) -> np.ndarray:
    h, w = arr.shape
    m = max(h, w)
    if m <= long_side:
        return arr
    scale = long_side / m
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    img = Image.fromarray(arr.astype(np.uint8) if arr.dtype != np.uint8 else arr)
    return np.array(img.resize((new_w, new_h), Image.BILINEAR), dtype=arr.dtype)


def run_one(image, gt, sigma, lambda_data, lambda_smooth, solver):
    t0 = time.time()
    res = segment(
        image,
        sigma=sigma,
        lambda_data=lambda_data,
        lambda_smooth=lambda_smooth,
        solver=solver,
    )
    dt = time.time() - t0

    mask = res.mask
    dice = iou = None
    if gt is not None:
        # auto-flip if labels swapped (so quality compare is fair)
        d1 = dice_score(mask, gt)
        d2 = dice_score(1 - mask, gt)
        if d2 > d1:
            mask = 1 - mask
            dice = d2
        else:
            dice = d1
        iou = iou_score(mask, gt)
    return {
        "solver": solver,
        "time": dt,
        "flow": res.max_flow_value,
        "fg_pixels": int(mask.sum()),
        "dice": dice,
        "iou": iou,
        "mask": mask,
    }


def fmt(x, w=10, prec=4):
    if x is None:
        return f"{'-':>{w}}"
    if isinstance(x, float):
        return f"{x:>{w}.{prec}f}"
    return f"{str(x):>{w}}"


SOLVER_LIST = ("ek", "dinic", "bk")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--gt", default=None)
    p.add_argument("--resize", type=int, nargs="+", default=[None],
                   help="One or more long-side sizes to benchmark; omit for original size.")
    p.add_argument("--sigma", type=float, default=30.0)
    p.add_argument("--lambda-data", type=float, default=1.0)
    p.add_argument("--lambda-smooth", type=float, default=20.0)
    p.add_argument("--solvers", nargs="+", default=list(SOLVER_LIST),
                   choices=list(SOLVER_LIST),
                   help="Which solvers to benchmark (default: all three).")
    p.add_argument("--save-dir", default=None,
                   help="If given, save each solver's mask per size.")
    args = p.parse_args()

    img_full = load_gray(args.image)
    gt_full = load_mask(args.gt) if args.gt else None

    print(f"\nImage: {args.image}    original shape: {img_full.shape}")
    if gt_full is not None:
        print(f"GT:    {args.gt}")
    print(f"Params: sigma={args.sigma}  lambda_data={args.lambda_data}  "
          f"lambda_smooth={args.lambda_smooth}")
    print()
    header = (f"{'size':>9} | {'solver':>6} | {'time (s)':>10} | {'flow':>10} | "
              f"{'fg px':>8} | {'Dice':>8} | {'IoU':>8}")
    print(header)
    print("-" * len(header))

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for size in args.resize:
        if size is None:
            img = img_full
            gt = gt_full
        else:
            img = resize(img_full, size)
            gt = None
            if gt_full is not None:
                gt = resize(gt_full, size)
                gt = (gt > 0).astype(np.uint8)
        size_str = f"{img.shape[0]}x{img.shape[1]}"

        results = {}
        for solver in args.solvers:
            r = run_one(img, gt, args.sigma, args.lambda_data,
                        args.lambda_smooth, solver)
            results[solver] = r
            print(f"{size_str:>9} | {solver:>6} | "
                  f"{fmt(r['time'])} | {fmt(r['flow'])} | "
                  f"{fmt(r['fg_pixels'], w=8, prec=0)} | "
                  f"{fmt(r['dice'], w=8)} | {fmt(r['iou'], w=8)}")

        # cross-solver agreement check
        flows = [results[s]['flow'] for s in args.solvers]
        flow_match = all(abs(f - flows[0]) < 1e-4 for f in flows)
        masks = [results[s]['mask'] for s in args.solvers]
        max_disagree = 0
        for i in range(1, len(masks)):
            d = int(np.sum(masks[i] != masks[0]))
            if d > max_disagree:
                max_disagree = d

        # speedup vs slowest
        times = {s: results[s]['time'] for s in args.solvers}
        slowest = max(times.values())
        fastest = min(times.values())
        spd_str = "  ".join(f"{s}: {slowest/times[s]:.2f}x" for s in args.solvers)
        print(f"{'':>9}   ↳ flows match: {flow_match}    "
              f"max mask disagreement: {max_disagree} px    "
              f"speedup-vs-slowest: {spd_str}")

        if save_dir:
            for solver in args.solvers:
                Image.fromarray(results[solver]["mask"] * 255).save(
                    save_dir / f"mask_{solver}_{size_str}.png")
        print()


if __name__ == "__main__":
    main()