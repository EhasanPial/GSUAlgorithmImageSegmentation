"""End-to-end segmentation pipeline: FCM -> build graph -> max-flow -> mask."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .fcm import run_fcm, resolve_obj_bg_from_seeds
from .graph import build_graph
from .maxflow import Graph as EKGraph
from .bk import Graph as BKGraph


SOLVERS = {"ek": EKGraph, "bk": BKGraph}


@dataclass
class SegmentationResult:
    mask: np.ndarray           # (H, W) uint8, 1 = object, 0 = background
    U: np.ndarray              # (c, N) FCM memberships
    centers: np.ndarray        # (c,) cluster centers
    obj_idx: int
    bg_idx: int
    max_flow_value: float
    solver: str = "ek"


def segment(
    image: np.ndarray,
    n_clusters: int = 2,
    m: float = 2.0,
    sigma: float = 30.0,
    lambda_data: float = 1.0,
    lambda_smooth: float = 20.0,
    seed: int = 42,
    fg_seeds: Optional[np.ndarray] = None,
    bg_seeds: Optional[np.ndarray] = None,
    solver: str = "ek",
) -> SegmentationResult:
    """Segment a grayscale image using FCM + Graph Cut.

    Parameters
    ----------
    image : (H, W) grayscale image.
    n_clusters : FCM clusters (2 for binary segmentation).
    m : FCM fuzzifier.
    sigma : stddev for n-link Gaussian weight (intensity scale).
    lambda_data : weight of data term (from FCM).
    lambda_smooth : weight of smoothness term (n-links).
    fg_seeds : optional (H, W) bool/uint8 mask. True/1 = foreground-locked pixel.
    bg_seeds : optional (H, W) bool/uint8 mask. True/1 = background-locked pixel.
    solver : 'ek' (Edmonds-Karp) or 'bk' (Boykov-Kolmogorov).
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {image.shape}")
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {list(SOLVERS)}; got {solver!r}")

    H, W = image.shape

    # 1) FCM memberships
    U, centers, obj_idx, bg_idx = run_fcm(
        image, n_clusters=n_clusters, m=m, seed=seed
    )

    # 1b) If seeds provided, override obj/bg cluster assignment using them.
    if fg_seeds is not None or bg_seeds is not None:
        fg_flat = (fg_seeds.astype(bool).ravel()
                   if fg_seeds is not None else np.zeros(H * W, dtype=bool))
        bg_flat = (bg_seeds.astype(bool).ravel()
                   if bg_seeds is not None else np.zeros(H * W, dtype=bool))
        obj_idx, bg_idx = resolve_obj_bg_from_seeds(U, fg_flat, bg_flat)

    # 2) Build graph using the chosen solver class
    g, source, sink = build_graph(
        image, U, obj_idx, bg_idx,
        sigma=sigma,
        lambda_data=lambda_data,
        lambda_smooth=lambda_smooth,
        fg_seeds=fg_seeds,
        bg_seeds=bg_seeds,
        graph_cls=SOLVERS[solver],
    )

    # 3) Max-flow
    flow_val = g.max_flow(source, sink)

    # 4) Min-cut: source-side nodes are labeled as OBJECT
    source_side = g.min_cut_source_side(source)
    mask = np.zeros(H * W, dtype=np.uint8)
    for i in range(H * W):
        if i in source_side:
            mask[i] = 1
    mask = mask.reshape(H, W)

    return SegmentationResult(
        mask=mask,
        U=U,
        centers=centers,
        obj_idx=obj_idx,
        bg_idx=bg_idx,
        max_flow_value=flow_val,
        solver=solver,
    )


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice similarity coefficient between two binary masks."""
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(inter / union)