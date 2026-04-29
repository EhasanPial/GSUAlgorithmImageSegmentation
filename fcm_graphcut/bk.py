"""Boykov-Kolmogorov (BK) max-flow / min-cut, from scratch.

Reference:
    Y. Boykov, V. Kolmogorov,
    "An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy
     Minimization in Vision," PAMI 2004.

Public API matches `maxflow.Graph`:
    g = Graph(n_nodes)
    g.add_edge(u, v, cap_uv, cap_vu=0.0)
    flow = g.max_flow(source, sink)
    src_side = g.min_cut_source_side(source)

Why a separate file?
    Edmonds-Karp restarts BFS from scratch every iteration -> O(V*E) per push.
    BK maintains TWO search trees rooted at S and T, reuses them across
    augmentations, and only locally repairs the trees when edges saturate.
    On image grids (lots of locality, short augmenting paths) this is
    typically 5-10x faster in practice than Edmonds-Karp.

The three phases (looped until done):
    1. GROWTH   : grow S-tree and T-tree alternately along non-saturated
                  edges until they touch -> augmenting path found.
    2. AUGMENT  : push bottleneck flow along the path; saturate edges;
                  any node whose parent edge got saturated becomes an
                  "orphan" (its subtree is detached from its root).
    3. ADOPT    : try to reconnect each orphan to its tree (find a new
                  valid parent). If impossible, the orphan is freed and
                  its children become orphans too.
    Loop ends when growth phase can't extend either tree.
"""
from __future__ import annotations

from collections import deque
from typing import List, Set, Optional


# Tree membership labels for each node
FREE = 0    # not in any tree yet
SOURCE = 1  # in S-tree
SINK = 2    # in T-tree


class Graph:
    """Directed graph with residual capacities for max-flow (BK solver)."""

    def __init__(self, n_nodes: int):
        self.n = n_nodes
        # adjacency: head[v] = list of edge IDs originating at v
        self.head: List[List[int]] = [[] for _ in range(n_nodes)]
        # parallel arrays for edges (same layout as Edmonds-Karp file)
        self.to: List[int] = []
        self.cap: List[float] = []

        # BK-specific per-node state
        self.tree: List[int] = [FREE] * n_nodes      # FREE / SOURCE / SINK
        self.parent_edge: List[int] = [-1] * n_nodes  # edge id that links node to its parent, -1 if root or free
        self.dist: List[int] = [0] * n_nodes          # distance to root (used to break cycles in adoption)
        self.timestamp: List[int] = [0] * n_nodes     # iteration count when dist was last verified
        self.global_time: int = 0

        # Active and orphan lists
        self._active: deque = deque()
        self._in_active: List[bool] = [False] * n_nodes
        self._orphans: deque = deque()

        self.source: int = -1
        self.sink: int = -1

    # ---- graph construction (identical to Edmonds-Karp version) ----

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

    # ---- helpers ----

    def _set_active(self, v: int) -> None:
        if not self._in_active[v]:
            self._in_active[v] = True
            self._active.append(v)

    def _residual_cap(self, eid: int, from_node: int) -> float:
        """Residual capacity of edge `eid` *in the direction relevant for tree
        membership of `from_node`*. For S-tree we walk along forward dir, so we
        want forward residual; for T-tree we walk along reverse direction."""
        if self.tree[from_node] == SOURCE:
            return self.cap[eid]            # going forward
        else:
            return self.cap[eid ^ 1]        # going backward (use the pair's capacity)

    # ---- BK three phases ----

    def _grow(self) -> int:
        """GROWTH phase. Returns edge id that connects the two trees, or -1."""
        while self._active:
            p = self._active[0]
            if self.tree[p] == FREE:
                # may have been freed during adoption
                self._active.popleft()
                self._in_active[p] = False
                continue

            for eid in self.head[p]:
                # capacity in the "tree-walking" direction
                if self.tree[p] == SOURCE:
                    cap_here = self.cap[eid]
                else:
                    cap_here = self.cap[eid ^ 1]
                if cap_here <= 1e-12:
                    continue

                q = self.to[eid]
                if self.tree[q] == FREE:
                    # add q to p's tree as p's child
                    self.tree[q] = self.tree[p]
                    self.parent_edge[q] = eid ^ 1   # edge from q's perspective points back to p
                    self.dist[q] = self.dist[p] + 1
                    self.timestamp[q] = self.timestamp[p]
                    self._set_active(q)
                elif self.tree[q] != self.tree[p]:
                    # opposite tree -> path found
                    # return the edge that crosses (always treat as edge
                    # going from S-tree node to T-tree node)
                    if self.tree[p] == SOURCE:
                        return eid
                    else:
                        return eid ^ 1

            # node p exhausted -> deactivate
            self._active.popleft()
            self._in_active[p] = False
        return -1

    def _augment(self, connecting_edge: int) -> float:
        """AUGMENT phase along the path implied by parent_edges + connecting_edge.

        connecting_edge goes from an S-tree node `a` to a T-tree node `b`
        (forward direction in the original graph)."""
        # 1) find bottleneck
        # walk from a back to source along S-tree (parent_edge points to edge
        # from child to parent, i.e. the reverse edge w.r.t. tree growth)
        bottleneck = self.cap[connecting_edge]

        # walk S-side: from `a` up to source
        a = self.to[connecting_edge ^ 1]   # tail of connecting edge (S-tree node)
        v = a
        while v != self.source:
            eid_to_parent = self.parent_edge[v]   # v -> parent(v) in tree-walk dir;
            # but parent_edge stores the edge from v's perspective, which points
            # *back* to parent. The "downward" forward residual for the path
            # S -> ... -> v is on the parent->v direction = (eid_to_parent ^ 1).
            forward_to_v = eid_to_parent ^ 1
            bottleneck = min(bottleneck, self.cap[forward_to_v])
            v = self.to[eid_to_parent]   # move to parent

        # walk T-side: from `b` down to sink
        b = self.to[connecting_edge]
        v = b
        while v != self.sink:
            eid_to_parent = self.parent_edge[v]
            # for T-tree, the path direction we send flow along is v -> parent(v),
            # which is exactly the forward direction of `eid_to_parent`.
            bottleneck = min(bottleneck, self.cap[eid_to_parent])
            v = self.to[eid_to_parent]

        # 2) push flow & detect orphans
        # connecting edge
        self.cap[connecting_edge] -= bottleneck
        self.cap[connecting_edge ^ 1] += bottleneck

        # S-side
        v = a
        while v != self.source:
            eid_to_parent = self.parent_edge[v]
            forward_to_v = eid_to_parent ^ 1
            self.cap[forward_to_v] -= bottleneck
            self.cap[eid_to_parent] += bottleneck
            if self.cap[forward_to_v] <= 1e-12:
                # parent->v edge saturated: v is now an orphan
                self.parent_edge[v] = -1
                self._orphans.append(v)
            v = self.to[eid_to_parent]

        # T-side
        v = b
        while v != self.sink:
            eid_to_parent = self.parent_edge[v]
            self.cap[eid_to_parent] -= bottleneck
            self.cap[eid_to_parent ^ 1] += bottleneck
            if self.cap[eid_to_parent] <= 1e-12:
                self.parent_edge[v] = -1
                self._orphans.append(v)
            v = self.to[eid_to_parent]

        return bottleneck

    def _origin(self, v: int) -> int:
        """Walk up parents to find which root a tree node ultimately reaches.
        Returns the root node (SOURCE or SINK) if connected, else -1.
        Also fills self.dist / self.timestamp along the way (path compression)."""
        # collect path
        path = []
        cur = v
        while True:
            if self.timestamp[cur] == self.global_time:
                # already validated this iter; we know its dist is correct
                root = self.source if self.tree[cur] == SOURCE else self.sink
                if cur == root:
                    base_dist = 0
                else:
                    # cur's distance is already computed
                    base_dist = self.dist[cur]
                # propagate
                d = base_dist + 1
                for node in reversed(path):
                    self.dist[node] = d
                    self.timestamp[node] = self.global_time
                    d += 1
                return root
            if cur == self.source:
                root = self.source
                break
            if cur == self.sink:
                root = self.sink
                break
            if self.parent_edge[cur] == -1:
                # broken chain -> not connected to root
                for node in path:
                    self.dist[node] = -1
                return -1
            path.append(cur)
            cur = self.to[self.parent_edge[cur]]

        # mark cur (= root) as validated
        self.dist[cur] = 0
        self.timestamp[cur] = self.global_time
        d = 1
        for node in reversed(path):
            self.dist[node] = d
            self.timestamp[node] = self.global_time
            d += 1
        return root

    def _adopt(self) -> None:
        """ADOPTION phase. Try to give each orphan a new parent in its tree;
        if impossible, free the orphan and orphan its children in turn."""
        while self._orphans:
            v = self._orphans.popleft()
            if self.tree[v] == FREE:
                continue

            tree_v = self.tree[v]
            best_parent = -1
            best_dist = float("inf")

            # try to find a same-tree neighbor that has a valid root path
            for eid in self.head[v]:
                # we need residual capacity in the direction *parent -> v*
                # (because we'll use this edge to push flow downward through v)
                if tree_v == SOURCE:
                    # v in S-tree: parent->v means edge eid^1 points from neighbor
                    # to v, i.e. neighbor q sends flow forward to v on eid^1.
                    # residual cap on q->v = cap[eid^1].
                    cap_in = self.cap[eid ^ 1]
                else:
                    # v in T-tree: parent q is downstream, edge from v->q is eid;
                    # we push flow v->q which uses cap[eid].
                    # But for adoption we need residual on q -> v in *tree-growth*
                    # sense, which for T-tree is the reverse direction = cap[eid].
                    cap_in = self.cap[eid]
                if cap_in <= 1e-12:
                    continue

                q = self.to[eid]
                if self.tree[q] != tree_v:
                    continue
                # check q has a valid path to its root
                root = self._origin(q)
                expected_root = self.source if tree_v == SOURCE else self.sink
                if root != expected_root:
                    continue
                if self.dist[q] < best_dist:
                    best_dist = self.dist[q]
                    best_parent = eid   # eid stores edge v->q; from v's perspective parent_edge[v] = eid
                    # Wait: parent_edge convention says parent_edge[v] = edge from v to parent.
                    # Edge `eid` in head[v] does go v->q (q = self.to[eid]). Good.

            if best_parent != -1:
                self.parent_edge[v] = best_parent
                self.dist[v] = best_dist + 1
                self.timestamp[v] = self.global_time
            else:
                # free v and orphan its children
                # 1) make all neighboring same-tree nodes active (they may want
                #    to grow back into v later)
                for eid in self.head[v]:
                    q = self.to[eid]
                    if self.tree[q] == tree_v:
                        # if q's parent IS v, q becomes an orphan
                        pe = self.parent_edge[q]
                        if pe != -1 and self.to[pe] == v:
                            self.parent_edge[q] = -1
                            self._orphans.append(q)
                        # otherwise activate q (it might be able to absorb v's
                        # neighbors later)
                        # check residual in growth direction
                        if tree_v == SOURCE:
                            cap_growth = self.cap[eid ^ 1]   # q -> v forward
                        else:
                            cap_growth = self.cap[eid]
                        if cap_growth > 1e-12:
                            self._set_active(q)
                self.tree[v] = FREE
                self.parent_edge[v] = -1
                # v is no longer in any tree; remove from active if there
                # (we just leave the stale flag; _grow checks tree[v])

    # ---- driver ----

    def max_flow(self, s: int, t: int) -> float:
        """Compute max-flow from s to t using BK. Returns total flow."""
        if s == t:
            return 0.0
        self.source = s
        self.sink = t

        # reset state (in case max_flow is called twice)
        self.tree = [FREE] * self.n
        self.parent_edge = [-1] * self.n
        self.dist = [0] * self.n
        self.timestamp = [0] * self.n
        self.global_time = 0
        self._active = deque()
        self._in_active = [False] * self.n
        self._orphans = deque()

        # initialize: source and sink are roots of their trees
        self.tree[s] = SOURCE
        self.tree[t] = SINK
        self._set_active(s)
        self._set_active(t)

        flow = 0.0
        while True:
            self.global_time += 1
            connecting_edge = self._grow()
            if connecting_edge == -1:
                break
            pushed = self._augment(connecting_edge)
            flow += pushed
            self._adopt()

        return flow

    # ---- min-cut extraction ----

    def min_cut_source_side(self, s: int) -> Set[int]:
        """Return set of nodes reachable from s in the residual graph after
        max_flow. With BK, source-side nodes are exactly nodes labeled SOURCE
        in `self.tree`. We BFS to be robust against any FREE leftovers."""
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