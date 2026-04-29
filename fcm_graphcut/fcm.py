"""Fuzzy C-Means clustering for pixel memberships.

Uses scikit-fuzzy's cmeans. Returns membership matrix U of shape (c, N)
where U[j, i] is the membership of pixel i in cluster j and columns sum to 1.
"""
from __future__ import annotations

import numpy as np
import skfuzzy as fuzz


def run_fcm(
    image: np.ndarray,
    n_clusters: int = 2,
    m: float = 2.0,
    error: float = 1e-5,
    max_iter: int = 200,
    seed: int = 42,
):
    """Run FCM on a 2D grayscale image.

    Parameters
    ----------
    image : (H, W) float array, values in any range (typically 0-255).
    n_clusters : number of clusters (2 for object/background).
    m : fuzzifier, m > 1 (2.0 is standard).

    Returns
    -------
    U : (c, N) membership matrix, N = H*W.
    centers : (c,) cluster centers sorted ascending by intensity.
    obj_idx : int, index of the "object" cluster (brightest center by default).
    bg_idx  : int, index of the "background" cluster (darkest center).
    """
    H, W = image.shape
    data = image.reshape(1, -1).astype(np.float64)  # shape (1, N)

    centers, U, _, _, _, _, _ = fuzz.cluster.cmeans(
        data,
        c=n_clusters,
        m=m,
        error=error,
        maxiter=max_iter,
        init=None,
        seed=seed,
    )

    # Sort clusters by center intensity: darkest = background, brightest = object
    centers = centers.flatten()
    order = np.argsort(centers)
    centers_sorted = centers[order]
    U_sorted = U[order, :]

    bg_idx = 0                 # darkest
    obj_idx = n_clusters - 1   # brightest
    return U_sorted, centers_sorted, obj_idx, bg_idx


def resolve_obj_bg_from_seeds(
    U: np.ndarray,
    fg_seeds_flat: np.ndarray,
    bg_seeds_flat: np.ndarray,
):
    """Decide which cluster is OBJECT and which is BACKGROUND using user seeds.

    Strategy: for each cluster j, sum its memberships over FG seeds vs BG seeds.
    The cluster with higher FG mass is the object cluster.

    Parameters
    ----------
    U : (c, N) membership matrix.
    fg_seeds_flat : (N,) bool array, True at foreground seed pixels.
    bg_seeds_flat : (N,) bool array, True at background seed pixels.

    Returns
    -------
    obj_idx, bg_idx
    """
    c = U.shape[0]
    fg_score = U[:, fg_seeds_flat].sum(axis=1) if fg_seeds_flat.any() else np.zeros(c)
    bg_score = U[:, bg_seeds_flat].sum(axis=1) if bg_seeds_flat.any() else np.zeros(c)

    if fg_seeds_flat.any() and bg_seeds_flat.any():
        # Cluster most "owned" by FG seeds vs by BG seeds
        obj_idx = int(np.argmax(fg_score - bg_score))
        bg_idx = int(np.argmax(bg_score - fg_score))
        if obj_idx == bg_idx:
            # tie-break: pick the second-best for bg
            order = np.argsort(bg_score - fg_score)[::-1]
            bg_idx = int(order[1]) if len(order) > 1 else (1 - obj_idx)
    elif fg_seeds_flat.any():
        obj_idx = int(np.argmax(fg_score))
        bg_idx = int(np.argmin(fg_score))  # cluster least like FG
    elif bg_seeds_flat.any():
        bg_idx = int(np.argmax(bg_score))
        obj_idx = int(np.argmin(bg_score))
    else:
        # no seeds: fall back to brightness convention
        obj_idx = U.shape[0] - 1
        bg_idx = 0
    return obj_idx, bg_idx