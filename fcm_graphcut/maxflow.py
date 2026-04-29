"""From-scratch max-flow / min-cut implementation.

Uses Edmonds-Karp (BFS-based augmenting paths) on an adjacency-list graph
with residual capacities. After max-flow, `min_cut_source_side` returns
the set of nodes reachable from the source in the residual graph — these
are the nodes on the source side of the min cut.

Edges are stored in pairs (forward + reverse) so that augmenting along
`e` updates capacity of `e^1` (its pair) automatically.

For binary image segmentation this is fast enough on small-to-medium
images (a 5x5 is trivial; 256x256 works in seconds). For very large
images, swap this for Boykov-Kolmogorov; the Graph API stays the same.
"""
from __future__ import annotations

from collections import deque
from typing import List, Set


class Graph:
    """Directed graph with residual capacities for max-flow."""

    def __init__(self, n_nodes: int):
        self.n = n_nodes
        # head[v] = list of edge indices starting at v
        self.head: List[List[int]] = [[] for _ in range(n_nodes)]
        # parallel arrays for edges
        self.to: List[int] = []
        self.cap: List[float] = []

    def add_edge(self, u: int, v: int, cap_uv: float, cap_vu: float = 0.0) -> int:
        """Add a pair of edges (u->v, v->u). Returns index of forward edge."""
        idx = len(self.to)
        # forward
        self.to.append(v)
        self.cap.append(float(cap_uv))
        self.head[u].append(idx)
        # reverse (its pair is idx, so idx^1 gives back the forward one)
        self.to.append(u)
        self.cap.append(float(cap_vu))
        self.head[v].append(idx + 1)
        return idx

    # ---- max-flow (Edmonds-Karp) ----

    def _bfs(self, s: int, t: int, parent_edge: List[int]) -> bool:
        for i in range(self.n):
            parent_edge[i] = -1
        parent_edge[s] = -2  # mark visited
        q = deque([s])
        while q:
            u = q.popleft()
            for eid in self.head[u]:
                v = self.to[eid]
                if parent_edge[v] == -1 and self.cap[eid] > 1e-12:
                    parent_edge[v] = eid
                    if v == t:
                        return True
                    q.append(v)
        return False

    def max_flow(self, s: int, t: int) -> float:
        flow = 0.0
        parent_edge = [-1] * self.n
        while self._bfs(s, t, parent_edge):
            # find bottleneck
            bottleneck = float("inf")
            v = t
            while v != s:
                eid = parent_edge[v]
                bottleneck = min(bottleneck, self.cap[eid])
                v = self.to[eid ^ 1]  # tail of forward edge = head of its pair
            # augment
            v = t
            while v != s:
                eid = parent_edge[v]
                self.cap[eid] -= bottleneck
                self.cap[eid ^ 1] += bottleneck
                v = self.to[eid ^ 1]
            flow += bottleneck
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