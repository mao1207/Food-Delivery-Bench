# Base/Graph.py
import math
from collections import defaultdict
from typing import Dict, Any, Optional, Tuple, List, Set

from Base.Types import Vector, Node, Edge


# ---------- 几何 ----------
def _proj_with_raw(p: Vector, a: Vector, b: Vector):
    """返回 (proj_raw, t_raw, proj_seg, t, da, db, seg_len)"""
    abx, aby = (b.x - a.x), (b.y - a.y)
    apx, apy = (p.x - a.x), (p.y - a.y)
    ab_len2 = abx * abx + aby * aby
    if ab_len2 <= 1e-9:
        proj_raw = Vector(a.x, a.y)
        return proj_raw, 0.0, proj_raw, 0.0, 0.0, 0.0, 0.0
    t_raw = (apx * abx + apy * aby) / ab_len2
    proj_raw = Vector(a.x + abx * t_raw, a.y + aby * t_raw)
    t = max(0.0, min(1.0, t_raw))
    proj_seg = Vector(a.x + abx * t, a.y + aby * t)
    da = math.hypot(proj_seg.x - a.x, proj_seg.y - a.y)
    db = math.hypot(proj_seg.x - b.x, proj_seg.y - b.y)
    seg_len = math.sqrt(ab_len2)
    return proj_raw, t_raw, proj_seg, t, da, db, seg_len


def project_point_to_segment(p: Vector, a: Vector, b: Vector):
    """给 Map 用的薄封：返回 (proj_raw, da_to_A_along_line, db_to_B_along_line, seg_len)。"""
    proj_raw, _t_raw, proj_seg, _t, da, db, seg_len = _proj_with_raw(p, a, b)
    # 这里的 da/db 用“线段内垂足”的几何距离，Map 侧只用 proj。
    return proj_raw, da, db, seg_len


# ---------- Graph ----------
class Graph:
    def __init__(self):
        self.nodes: Set[Node] = set()
        self.edges: Set[Edge] = set()
        self.adjacency_list: Dict[Node, List[Node]] = defaultdict(list)
        self.edge_meta: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # ---- 基础 ----
    def add_node(self, node: Node):
        self.nodes.add(node)

    def add_edge(self, edge: Edge):
        if edge.node1 is edge.node2:
            return
        self.edges.add(edge)
        if edge.node2 not in self.adjacency_list[edge.node1]:
            self.adjacency_list[edge.node1].append(edge.node2)
        if edge.node1 not in self.adjacency_list[edge.node2]:
            self.adjacency_list[edge.node2].append(edge.node1)

    def connected(self, a: Node, b: Node) -> bool:
        return b in self.adjacency_list[a]

    def _edge_key(self, a: Node, b: Node) -> Tuple[str, str]:
        ka, kb = str(a), str(b)
        return (ka, kb) if ka < kb else (kb, ka)

    def set_edge_meta(self, a: Node, b: Node, meta: Dict[str, Any]):
        self.edge_meta[self._edge_key(a, b)] = dict(meta)

    def get_edge_meta(self, a: Node, b: Node) -> Optional[Dict[str, Any]]:
        return self.edge_meta.get(self._edge_key(a, b))

    # ---- 工具 ----
    def find_existing_node_at(self, pos: Vector, eps_cm: float = 80.0) -> Optional[Node]:
        px, py = float(pos.x), float(pos.y)
        thr2 = eps_cm * eps_cm
        best = None
        best_d2 = None
        for nd in self.nodes:
            dx = float(nd.position.x) - px
            dy = float(nd.position.y) - py
            d2 = dx * dx + dy * dy
            if d2 <= thr2 and (best is None or d2 < best_d2):
                best, best_d2 = nd, d2
        return best

    def _iter_road_edges_only(self):
        for e in list(self.edges):
            meta = self.get_edge_meta(e.node1, e.node2) or {}
            if meta.get("kind") not in ("road", "crosswalk", "endcap"):
                continue
            # 避免把 dock_link 的端点边当路
            if getattr(e.node1, "type", "") == "dock" or getattr(e.node2, "type", "") == "dock":
                continue
            yield e

    # ---- 吸附 ----
    def snap_to_nearest_edge(self, x: float, y: float) -> Optional[Dict[str, Any]]:
        """只在 road 边内用“线段距离”吸附；同时返回 raw/clamped 两套投影。"""
        p = Vector(x, y)
        best = None
        for e in self._iter_road_edges_only():
            a, b = e.node1.position, e.node2.position
            proj_raw, t_raw, proj_seg, t, da, db, _ = _proj_with_raw(p, a, b)
            d = p.distance(proj_seg)  # 线段距离
            if (best is None) or (d < best["dist"] - 1e-6):
                best = {
                    "edge": e,
                    "a": e.node1, "b": e.node2,
                    "proj": proj_seg,       # clamp 后
                    "proj_raw": proj_raw,   # 可能在延长线
                    "t": t, "t_raw": t_raw,
                    "da": da, "db": db,
                    "dist": d
                }
        return best

    # ---- 在边上插点（段内切边；延长线短接）----
    def ensure_node_on_edge_at(
        self,
        edge: Edge,
        proj_raw: Vector,
        snap_cm: float = 100.0,
        max_extend_cm: float = 1500.0,
    ) -> Node:
        a, b = edge.node1, edge.node2
        ax, ay = a.position.x, a.position.y
        bx, by = b.position.x, b.position.y
        px, py = proj_raw.x, proj_raw.y

        old_meta = self.get_edge_meta(a, b) or {}

        # 计算 t_raw 判断是否在延长线
        abx, aby = (bx - ax), (by - ay)
        apx, apy = (px - ax), (py - ay)
        ab_len2 = abx * abx + aby * aby
        t_raw = (apx * abx + apy * aby) / (ab_len2 if ab_len2 > 1e-9 else 1.0)

        # —— 延长线：仅允许短接，类型= dock + dock_link
        if t_raw <= 0.0 or t_raw >= 1.0:
            from_node = a if t_raw <= 0.0 else b
            extend_len = math.hypot(px - from_node.position.x, py - from_node.position.y)
            new_node = Node(Vector(px, py), "dock")
            self.add_node(new_node)
            self.add_edge(Edge(from_node, new_node))
            self.set_edge_meta(from_node, new_node, {"kind": "dock_link", "name": ""})
            return new_node

        # —— 段内：切边
        new_node = Node(Vector(px, py), "normal")
        self.add_node(new_node)

        # 移除原边与邻接
        while edge in self.edges:
            self.edges.remove(edge)
        while b in self.adjacency_list.get(a, []):
            self.adjacency_list[a].remove(b)
        while a in self.adjacency_list.get(b, []):
            self.adjacency_list[b].remove(a)
        if old_meta:
            self.edge_meta.pop(self._edge_key(a, b), None)

        # 加两条新边并继承 meta
        self.add_edge(Edge(a, new_node))
        self.add_edge(Edge(new_node, b))
        if old_meta:
            self.set_edge_meta(a, new_node, old_meta)
            self.set_edge_meta(new_node, b, old_meta)

        return new_node

    # ---- 最短路 ----
    def _w(self, u: Node, v: Node) -> float:
        return u.position.distance(v.position)

    def shortest_path_nodes(self, start: Node, target: Node):
        import heapq
        dist: Dict[Node, float] = {start: 0.0}
        prev: Dict[Node, Node] = {}
        pq = [(0.0, id(start), start)]
        seen = set()

        while pq:
            du, _, u = heapq.heappop(pq)
            if u in seen:
                continue
            seen.add(u)
            if u == target:
                break
            for v in self.adjacency_list[u]:
                alt = du + self._w(u, v)
                if (v not in dist) or (alt < dist[v] - 1e-6):
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(pq, (alt, id(v), v))

        if target not in dist:
            return [], float("inf")

        path = []
        cur = target
        while cur != start:
            path.append(cur)
            cur = prev[cur]
        path.append(start)
        path.reverse()
        return path, float(dist[target])

    def shortest_path_xy_to_node(self, x: float, y: float, target: Node):
        snap = self.snap_to_nearest_edge(x, y)
        if snap is None:
            return [], float("inf"), None
        a, b = snap["a"], snap["b"]
        da, db = snap["da"], snap["db"]

        import heapq
        dist: Dict[Node, float] = {a: da, b: db}
        prev: Dict[Node, Node] = {}
        pq = [(da, id(a), a), (db, id(b), b)]
        seen = set()
        start_choice = None

        while pq:
            du, _, u = heapq.heappop(pq)
            if u in seen:
                continue
            seen.add(u)
            if start_choice is None and (u is a or u is b):
                start_choice = u
            if u == target:
                break
            for v in self.adjacency_list[u]:
                alt = du + self._w(u, v)
                if (v not in dist) or (alt < dist[v] - 1e-6):
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(pq, (alt, id(v), v))

        if target not in dist:
            return [], float("inf"), start_choice

        path = []
        cur = target
        while cur in prev:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path, float(dist[target]), start_choice

    # ---- 拷贝 ----
    def copy(self) -> "Graph":
        g = Graph()
        # 原始节点对象可以复用（寻路不会改原节点）
        for n in self.nodes:
            g.add_node(n)
        for e in self.edges:
            a, b = e.node1, e.node2
            ee = Edge(a, b)
            g.add_edge(ee)
            meta = self.get_edge_meta(a, b)
            if meta:
                g.set_edge_meta(a, b, meta)
        return g
