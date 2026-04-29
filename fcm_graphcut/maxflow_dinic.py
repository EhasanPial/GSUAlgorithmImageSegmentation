"""Dinic's max-flow / min-cut algorithm, from scratch.

Reference:
    E. A. Dinic, "Algorithm for solution of a problem of maximum flow in
    a network with power estimation," Soviet Math. Dokl., 1970.

Public API matches `maxflow.Graph` and `maxflow_bk.Graph`:
    g = Graph(n_nodes)
    g.add_edge(u, v, cap_uv, cap_vu=0.0)
    flow = g.max_flow(source, sink)
    src_side = g.min_cut_source_side(source)

Two phases per iteration, looped until no augmenting path remains:

    1. BFS levels    : compute shortest distance (in edges) from source to
                       every node in the residual graph. If sink is
                       unreachable, we're done.

    2. Blocking flow : DFS many augmenting paths along the LEVEL graph
                       (only edges level[u]+1 == level[v] are allowed).
                       Push flow on each path, saturating at least one
                       edge each time. Use a per-node "current edge"
                       pointer to skip already-explored edges (the
                       crucial speedup vs Edmonds-Karp).

Complexity:  O(V^2 * E) in general, O(E * sqrt(V)) for unit-capacity
graphs. For grid graphs typical of image segmentation, this is much
faster than Edmonds-Karp because each BFS yields multiple augmenting
paths instead of just one.
"""
from __future__ import annotations

import sys
from collections import deque
from typing import List, Set


# Image grids can produce DFS chains hundreds of thousands long.
# Bump the recursion limit so DFS doesn't blow up on a 256x256 image.
if sys.getrecursionlimit() < 200_000:
    sys.setrecursionlimit(200_000)


class Graph:
    """Directed graph with residual capacities for max-flow (Dinic solver)."""

    def __init__(self, n_nodes: int):
        self.n = n_nodes
        # adjacency: head[v] = list of edge IDs originating at v
        self.head: List[List[int]] = [[] for _ in range(n_nodes)]
        # parallel arrays for edges (same layout as the EK and BK files)
        self.to: List[int] = []
        self.cap: List[float] = []

        # Dinic-specific per-node state
        self.level: List[int] = [-1] * n_nodes  # BFS distance from source
        self.iter_ptr: List[int] = [0] * n_nodes  # current-edge pointer for DFS

    # ---- graph construction (identical to the other solvers) ----

    def add_edge(self, u: int, v: int, cap_uv: float, cap_vu: float = 0.0) -> int:
        """Add forward edge u->v and reverse edge v->u. Returns forward edge id."""
        idx = len(self.to)
        # forward
        self.to.append(v)
        self.cap.append(float(cap_uv))
        self.head[u].append(idx)
        # reverse (pair = idx ^ 1)
        self.to.append(u)
        self.cap.append(float(cap_vu))
        self.head[v].append(idx + 1)
        return idx

    # ---- phase 1: build level graph via BFS ----

    def _bfs_levels(self, s: int, t: int) -> bool:
        """Compute level[v] = shortest residual distance s -> v.
        Returns True if sink is reachable (an augmenting path exists)."""
        self.level = [-1] * self.n
        self.level[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for eid in self.head[u]:
                v = self.to[eid]
                if self.level[v] == -1 and self.cap[eid] > 1e-12:
                    self.level[v] = self.level[u] + 1
                    q.append(v)
        return self.level[t] != -1

    # ---- phase 2: DFS blocking flow on the level graph ----

    def _dfs(self, u: int, t: int, pushed: float) -> float:
        """DFS from u toward t along level-graph edges. Push up to `pushed`
        units of flow. Returns actual flow pushed (0 if no path)."""
        if u == t:
            return pushed
        # iter_ptr[u] tells us which edge in head[u] to try next; we never
        # revisit earlier edges within this blocking-flow phase because
        # they were either saturated or proven dead-end.
        while self.iter_ptr[u] < len(self.head[u]):
            eid = self.head[u][self.iter_ptr[u]]
            v = self.to[eid]
            # only follow level-graph edges with residual capacity
            if self.cap[eid] > 1e-12 and self.level[v] == self.level[u] + 1:
                bottleneck = min(pushed, self.cap[eid])
                d = self._dfs(v, t, bottleneck)
                if d > 1e-12:
                    self.cap[eid] -= d
                    self.cap[eid ^ 1] += d
                    return d
                # No flow through v on this branch -> dead end; mark v
                # unreachable for the rest of this phase by knocking its
                # level down, so other DFS calls skip it.
                self.level[v] = -1
            self.iter_ptr[u] += 1
        return 0.0

    # ---- driver ----

    def max_flow(self, s: int, t: int) -> float:
        """Compute max-flow from s to t using Dinic's algorithm."""
        if s == t:
            return 0.0
        flow = 0.0
        while self._bfs_levels(s, t):
            # reset current-edge pointers each phase
            self.iter_ptr = [0] * self.n
            # repeatedly DFS augmenting paths until none remain in the
            # current level graph (this is the "blocking flow" step).
            while True:
                pushed = self._dfs(s, t, float("inf"))
                if pushed <= 1e-12:
                    break
                flow += pushed
        return flow

    # ---- min-cut extraction ----

    def min_cut_source_side(self, s: int) -> Set[int]:
        """Return set of nodes reachable from s in the residual graph
        *after* max_flow has been run. These are the nodes on the source
        side of the min cut."""
        visited = {s}
        q = deque([s])
        while q:
            u = q.popleft()
            for eid in self.head[u]:
                v = self.to[eid]
                if v not in visited and self.cap[eid] > 1e-12:
                    visited.add(v)
                    q.append(v)
        return visited