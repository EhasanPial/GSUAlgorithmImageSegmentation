"""Generate synthetic test images (image + ground-truth mask) for quick demos.

Usage:
    python -m fcm_graphcut.make_synthetic --out samples/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def make_circle(H: int = 128, W: int = 128, noise: float = 20.0,
                seed: int = 0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H / 2, W / 2
    r = min(H, W) * 0.28
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r

    img = np.where(mask, 200.0, 40.0)
    img = img + rng.normal(0, noise, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, mask.astype(np.uint8)


def make_two_blobs(H: int = 128, W: int = 160, noise: float = 25.0,
                   seed: int = 1):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    m1 = ((yy - H * 0.4) ** 2 + (xx - W * 0.3) ** 2) <= (min(H, W) * 0.18) ** 2
    m2 = ((yy - H * 0.6) ** 2 + (xx - W * 0.7) ** 2) <= (min(H, W) * 0.22) ** 2
    mask = m1 | m2
    img = np.where(mask, 210.0, 50.0) + rng.normal(0, noise, (H, W))
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, mask.astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="samples")
    p.add_argument("--noise", type=float, default=20.0)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    img, gt = make_circle(noise=args.noise)
    Image.fromarray(img).save(out / "circle.png")
    Image.fromarray(gt * 255).save(out / "circle_gt.png")

    img, gt = make_two_blobs(noise=args.noise)
    Image.fromarray(img).save(out / "blobs.png")
    Image.fromarray(gt * 255).save(out / "blobs_gt.png")

    print(f"wrote samples to {out.resolve()}")


if __name__ == "__main__":
    main()
