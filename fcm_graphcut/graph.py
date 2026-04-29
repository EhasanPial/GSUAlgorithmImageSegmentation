"""Build the flow graph for binary segmentation.

Graph has:
  - N pixel nodes (indices 0..N-1)
  - Source S = N, Sink T = N+1
  - t-links from S to each pixel (capacity = D_i(bg))
  - t-links from each pixel to T (capacity = D_i(obj))
  - n-links between 4-neighbors (capacity = w_ij, undirected → two directed edges)

Data cost uses log of FCM memberships, clipped to avoid log(0).
Smoothness uses Gaussian of intensity difference.
"""
from __future__ import annotations

from typing import Tuple, Type

import numpy as np

from .maxflow import Graph as EKGraph


EPS = 1e-10


def data_costs(
    U: np.ndarray,
    obj_idx: int,
    bg_idx: int,
    lambda_data: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute t-link capacities.

    Returns
    -------
    cap_s : (N,) capacity S -> pixel i = lambda * (-log u_{i,bg})
    cap_t : (N,) capacity pixel i -> T = lambda * (-log u_{i,obj})
    """
    u_obj = np.clip(U[obj_idx], EPS, 1.0)
    u_bg = np.clip(U[bg_idx], EPS, 1.0)
    cap_s = lambda_data * (-np.log(u_bg))
    cap_t = lambda_data * (-np.log(u_obj))
    return cap_s, cap_t


def smoothness_weight(i1: float, i2: float, sigma: float) -> float:
    d = i1 - i2
    return float(np.exp(-(d * d) / (2.0 * sigma * sigma)))


def build_graph(
    image: np.ndarray,
    U: np.ndarray,
    obj_idx: int,
    bg_idx: int,
    sigma: float = 30.0,
    lambda_data: float = 1.0,
    lambda_smooth: float = 1.0,
    fg_seeds: np.ndarray | None = None,
    bg_seeds: np.ndarray | None = None,
    inf_cap: float = 1e9,
    graph_cls: Type = EKGraph,
) -> Tuple[object, int, int]:
    """Construct a max-flow graph from an image + FCM memberships.

    Optional seed arrays (boolean, shape (H, W)) impose hard constraints:
      - FG seed: cap(S->i) = INF, cap(i->T) = 0  (pixel locked to OBJECT)
      - BG seed: cap(S->i) = 0,   cap(i->T) = INF (pixel locked to BACKGROUND)

    `graph_cls` lets the caller pick the solver implementation
    (Edmonds-Karp Graph or BK Graph). Both expose the same `add_edge`,
    `max_flow`, `min_cut_source_side` API.

    Returns (graph, source_id, sink_id).
    """
    H, W = image.shape
    N = H * W
    source = N
    sink = N + 1
    n_nodes = N + 2

    g = graph_cls(n_nodes)

    # --- t-links ---
    cap_s, cap_t = data_costs(U, obj_idx, bg_idx, lambda_data)
    flat = image.astype(np.float64).ravel()

    # Apply seed hard constraints, if any
    if fg_seeds is not None:
        fg_flat = fg_seeds.astype(bool).ravel()
        cap_s = np.where(fg_flat, inf_cap, cap_s)
        cap_t = np.where(fg_flat, 0.0, cap_t)
    if bg_seeds is not None:
        bg_flat = bg_seeds.astype(bool).ravel()
        cap_s = np.where(bg_flat, 0.0, cap_s)
        cap_t = np.where(bg_flat, inf_cap, cap_t)

    for i in range(N):
        g.add_edge(source, i, cap_s[i], 0.0)
        g.add_edge(i, sink, cap_t[i], 0.0)

    # --- n-links (4-neighborhood, undirected -> symmetric capacities) ---
    for r in range(H):
        for c in range(W):
            i = r * W + c
            # right neighbor
            if c + 1 < W:
                j = i + 1
                w = lambda_smooth * smoothness_weight(flat[i], flat[j], sigma)
                g.add_edge(i, j, w, w)
            # down neighbor
            if r + 1 < H:
                j = i + W
                w = lambda_smooth * smoothness_weight(flat[i], flat[j], sigma)
                g.add_edge(i, j, w, w)

    return g, source, sink