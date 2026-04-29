"""Visualization helpers: overlay, side-by-side panels."""
from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45,
                 color=(1.0, 0.0, 0.0)) -> np.ndarray:
    """Return RGB image with mask drawn in `color` on top of grayscale `image`."""
    if image.ndim == 2:
        rgb = np.stack([image] * 3, axis=-1).astype(np.float32)
    else:
        rgb = image.astype(np.float32).copy()
    if rgb.max() > 1.5:
        rgb /= 255.0

    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    m = (mask > 0).astype(np.float32)[..., None]
    out = rgb * (1 - alpha * m) + color_arr * (alpha * m)
    return np.clip(out, 0, 1)


def make_panel(
    image: np.ndarray,
    mask: np.ndarray,
    U: np.ndarray,
    obj_idx: int,
    gt: Optional[np.ndarray] = None,
    dice: Optional[float] = None,
    iou: Optional[float] = None,
) -> Figure:
    """Build a matplotlib figure with: input | FCM obj-membership | mask | overlay."""
    H, W = image.shape
    obj_map = U[obj_idx].reshape(H, W)

    n_cols = 4 if gt is None else 5
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")

    im = axes[1].imshow(obj_map, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("FCM object membership")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(mask, cmap="gray", vmin=0, vmax=1)
    title = "Segmentation"
    if dice is not None:
        title += f"\nDice={dice:.4f}"
    if iou is not None:
        title += f"  IoU={iou:.4f}"
    axes[2].set_title(title)
    axes[2].axis("off")

    axes[3].imshow(overlay_mask(image, mask))
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    if gt is not None:
        axes[4].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[4].set_title("Ground truth")
        axes[4].axis("off")

    fig.tight_layout()
    return fig
