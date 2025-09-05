# Base/Map.py
import json
import math
from collections import deque
from typing import List, Dict, Any, Optional, Tuple

from Base.Types import Vector, Node, Edge, Road
from Config import Config
from Base.Graph import Graph, project_point_to_segment  # 几何投影

# =============== 角度工具 ===============
def _angle_deg(vx: float, vy: float) -> float:
    ang = math.degrees(math.atan2(vy, vx))
    ang = (ang + 360.0) % 360.0
    if ang >= 180.0:
        ang -= 180.0
    return ang

def _fmt_m(cm: float, decimals: int = 2) -> str:
    return f"{float(cm)/100.0:.{decimals}f}m"

def _fmt_xy_m(x_cm: float, y_cm: float, decimals: int = 2) -> str:
    return f"({float(x_cm)/100.0:.{decimals}f}m, {float(y_cm)/100.0:.{decimals}f}m)"


class Map:
    NEAR_EPS = 100.0  # 1m
    DOOR_YAW_OFFSET_DEG = 90.0

    def __init__(self):
        # 两张图：完整路网（full） & 骨架（skel）
        self.graph_full = Graph()
        self.graph_skel = Graph()

        # 语义 POI/订单
        self.pois: List[Node] = []
        # 每项：{"node":<poi Node>, "dock_node":<Node>, "door_node":<Node|None>,
        #        "road_name":<str|None>, "center":(x,y), "building_box":{x,y,w,h,yaw,poi_type}|None}
        self.poi_meta: List[Dict[str, Any]] = []
        self._dock_nodes = set()
        self.order_meta: List[Dict[str, Any]] = []

        self._road_id = 0
        self._xwalk_id = 0

        self._door2poi: Dict[Node, Node] = {}

        # 稳定编号（按 JSON 读取顺序）
        self._poi_seq_counter: Dict[str, int] = {}

    # ---------------- 兼容层（转发到 graph_full） ----------------
    @property
    def nodes(self):
        return self.graph_full.nodes

    @property
    def edges(self):
        return self.graph_full.edges

    @property
    def adjacency_list(self):
        return self.graph_full.adjacency_list

    @property
    def edge_meta(self):
        return self.graph_full.edge_meta

    def add_node(self, node: Node):
        self.graph_full.add_node(node)

    def add_edge(self, edge: Edge):
        self.graph_full.add_edge(edge)

    def _connected(self, a: Node, b: Node) -> bool:
        return self.graph_full.connected(a, b)

    def _edge_key(self, a: Node, b: Node) -> Tuple[str, str]:
        return self.graph_full._edge_key(a, b)

    def _set_edge_meta(self, a: Node, b: Node, meta: Dict[str, Any]):
        self.graph_full.set_edge_meta(a, b, meta)

    def _get_edge_meta(self, a: Node, b: Node) -> Optional[Dict[str, Any]]:
        return self.graph_full.get_edge_meta(a, b)

    # ---------------- 几何 ----------------
    @staticmethod
    def _project_point_to_segment(p: Vector, a: Vector, b: Vector) -> Tuple[Vector, float]:
        """
        返回线段 a->b 上的最近点 proj（t 已 clamp 到 [0,1]），以及参数 t。
        用于 skeleton 路名时可避免落到延长线导致的误判。
        """
        ax, ay = float(a.x), float(a.y)
        bx, by = float(b.x), float(b.y)
        px, py = float(p.x), float(p.y)

        vx, vy = (bx - ax), (by - ay)
        den = vx * vx + vy * vy
        if den <= 1e-12:
            # 退化：a==b，直接返回端点
            return Vector(ax, ay), 0.0

        t = ((px - ax) * vx + (py - ay) * vy) / den
        # clamp 到线段
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        proj = Vector(ax + vx * t, ay + vy * t)
        return proj, t

    @staticmethod
    def _rotate(vec: Vector, ang_rad: float) -> Vector:
        c, s = math.cos(ang_rad), math.sin(ang_rad)
        return Vector(vec.x * c - vec.y * s, vec.x * s + vec.y * c)

    # ---------------- bbox / 吸附 ----------------
    def bbox(self) -> Tuple[float, float, float, float]:
        xs = [float(n.position.x) for n in self.nodes]
        ys = [float(n.position.y) for n in self.nodes]
        if not xs or not ys:
            return (-1000, 1000, -1000, 1000)
        return (min(xs), max(xs), min(ys), max(ys))

    def snap_to_nearest_edge(
        self, x: float, y: float, use_segment: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        吸附到最近的真实边（road/crosswalk/endcap），忽略辅助边。
        返回 {'edge','a','b','proj','da','db','t'}。

        - use_segment=False（默认）：主排序=点到“无限直线”的距离；平手打破=点到“线段”的距离。
        - use_segment=True：主排序=点到“线段”的距离；平手打破=点到“直线”的距离。
        """
        p = Vector(x, y)
        best = None
        best_key = None   # (primary, secondary)
        EPS = 1e-9

        for e in list(self.edges):
            a, b = e.node1, e.node2
            meta = self._get_edge_meta(a, b) or {}
            kind = (meta.get("kind") or "")
            if kind.startswith("aux_"):
                continue
            if kind not in ("road", "crosswalk", "endcap"):
                continue

            ax, ay = float(a.position.x), float(a.position.y)
            bx, by = float(b.position.x), float(b.position.y)
            px, py = float(p.x), float(p.y)

            vx, vy = (bx - ax), (by - ay)
            den = vx * vx + vy * vy
            if den <= 1e-12:
                continue  # 退化边，跳过

            # 原始直线参数 & 投影点（可能在延长线上）
            t_raw = ((px - ax) * vx + (py - ay) * vy) / den
            proj_line = Vector(ax + vx * t_raw, ay + vy * t_raw)

            # 线段截断投影
            t_seg = max(0.0, min(1.0, t_raw))
            proj_seg = Vector(ax + vx * t_seg, ay + vy * t_seg)

            # 两种距离（平方）
            d_line2 = (px - proj_line.x) ** 2 + (py - proj_line.y) ** 2
            d_seg2  = (px - proj_seg.x) ** 2  + (py - proj_seg.y) ** 2

            # 选择 key
            key = (d_seg2, d_line2) if use_segment else (d_line2, d_seg2)

            if (best_key is None or
                key[0] < best_key[0] - EPS or
                (abs(key[0] - best_key[0]) <= EPS and key[1] < best_key[1] - EPS)):
                best_key = key
                chosen_proj = proj_seg if use_segment else proj_line
                best = {
                    "edge": e,
                    "a": a, "b": b,
                    "proj": chosen_proj,           # 返回与模式匹配的投影
                    "da": a.position.distance(chosen_proj),
                    "db": b.position.distance(chosen_proj),
                    "t": float(t_raw),             # 保留原始直线参数 t_raw
                }
        return best



    # ---------------- 类型判断 ----------------
    @staticmethod
    def _t_of(nd: Node) -> str:
        return (getattr(nd, "type", "") or "").lower()

    def _is_roadpoint(self, nd: Node, include_docks: bool) -> bool:
        t = self._t_of(nd)
        if t in ("normal", "intersection"):
            if not include_docks and nd in self._dock_nodes:
                return False
            return True
        return False

    def _is_transparent(self, nd: Node) -> bool:
        t = self._t_of(nd)
        return (nd in self._dock_nodes) or (t in ("door", "bus_station", "charging_station"))

    # ---------------- 朝向/分组（只用 yaw） ----------------
    @staticmethod
    def _quantize_yaw_deg(yaw: float) -> float:
        """把 yaw 量化到 {0,90,180,270} 之一。"""
        a = ((float(yaw) % 360.0) + 360.0) % 360.0
        choices = [0.0, 90.0, 180.0, 270.0]
        return min(choices, key=lambda c: min(abs(c - a), 360.0 - abs(c - a)))

    @staticmethod
    def _unit_from_angle(angle_deg: float) -> Vector:
        """世界坐标：0→+x, 90→+y, 180/−180→−x, 270/−90→−y。"""
        rad = math.radians(angle_deg)
        return Vector(math.cos(rad), math.sin(rad))

    @staticmethod
    def _edge_group(a: Node, b: Node, tol_deg: float = 6.0) -> Optional[str]:
        """把边分到 'h' 或 'v'；若偏离水平/垂直超过 tol 则返回 None."""
        vx, vy = (b.position.x - a.position.x), (b.position.y - a.position.y)
        ang = abs(_angle_deg(vx, vy))  # [0,180]
        if min(ang, abs(ang - 180.0)) <= tol_deg:
            return "h"
        if abs(ang - 90.0) <= tol_deg:
            return "v"
        return None

    @staticmethod
    def _hv_group_from_yaw(yaw_q: float) -> str:
        """量化后 0/180 归 h，90/270 归 v。"""
        return "h" if yaw_q in (0.0, 180.0) else "v"

    # —— 按组吸附最近边（优先同组 h/v）
    def _snap_to_nearest_edge_with_group(
        self, x: float, y: float, group: Optional[str], use_segment: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        返回 {'edge','a','b','proj','da','db','t'}。
        - 仅考虑 kind ∈ {road, crosswalk, endcap} 的“真实边”，忽略 aux_*。
        - 若指定 group ∈ {'h','v'}，优先在该组内选；若找不到再全量兜底。
        - use_segment=False：主排序=点到“直线”距离，平手=点到“线段”距离；
        use_segment=True：主排序=点到“线段”距离，平手=点到“直线”距离。
        - 返回的 't' 为直线参数 t_raw（无界），'proj' 与模式匹配（线段/直线）。
        """
        def _best(prefer_group: Optional[str]) -> Optional[Dict[str, Any]]:
            p = Vector(x, y)
            best = None
            best_key = None  # (primary, secondary)
            EPS = 1e-9

            for e in list(self.edges):
                a, b = e.node1, e.node2
                meta = self._get_edge_meta(a, b) or {}
                kind = (meta.get("kind") or "")
                if kind.startswith("aux_"):
                    continue
                if kind not in ("road", "crosswalk", "endcap"):
                    continue

                # 组过滤（h/v）
                g = self._edge_group(a, b)
                if prefer_group is not None and g != prefer_group:
                    continue

                # 计算直线/线段两套投影
                ax, ay = float(a.position.x), float(a.position.y)
                bx, by = float(b.position.x), float(b.position.y)
                px, py = float(p.x), float(p.y)

                vx, vy = (bx - ax), (by - ay)
                den = vx * vx + vy * vy
                if den <= 1e-12:
                    continue  # 退化边

                t_raw = ((px - ax) * vx + (py - ay) * vy) / den
                proj_line = Vector(ax + vx * t_raw, ay + vy * t_raw)  # 直线投影
                t_seg = 0.0 if t_raw < 0.0 else (1.0 if t_raw > 1.0 else t_raw)
                proj_seg = Vector(ax + vx * t_seg, ay + vy * t_seg)   # 线段投影

                d_line2 = (px - proj_line.x) ** 2 + (py - proj_line.y) ** 2
                d_seg2  = (px - proj_seg.x)  ** 2 + (py - proj_seg.y)  ** 2

                key = (d_seg2, d_line2) if use_segment else (d_line2, d_seg2)
                if (best_key is None or
                    key[0] < best_key[0] - EPS or
                    (abs(key[0] - best_key[0]) <= EPS and key[1] < best_key[1] - EPS)):
                    best_key = key
                    chosen_proj = proj_seg if use_segment else proj_line
                    best = {
                        "edge": e,
                        "a": a, "b": b,
                        "proj": chosen_proj,
                        "da": a.position.distance(chosen_proj),
                        "db": b.position.distance(chosen_proj),
                        "t": float(t_raw),   # 无界直线参数
                    }
            return best

        ret = _best(group)
        if ret is not None:
            return ret
        return _best(None)


    # —— 最近骨干路名（按组 h/v）
    def _nearest_skeleton_road_name_by_group(self, px: float, py: float, group: Optional[str]) -> Optional[str]:
        p = Vector(px, py)
        best_name, best_d = None, None
        for e in self.graph_skel.edges:
            u, v = e.node1, e.node2
            meta = self.graph_skel.get_edge_meta(u, v) or {}
            if meta.get("kind") != "road":
                continue
            name = meta.get("name") or ""
            if not name:
                continue
            g = self._edge_group(u, v)
            if group is not None and g != group:
                continue
            proj, _ = self._project_point_to_segment(p, u.position, v.position)
            d = p.distance(proj)
            if (best_d is None) or (d < best_d - 1e-9):
                best_d, best_name = d, name
        # 找不到同组则退化全量
        if best_name is None and group is not None:
            return self._nearest_skeleton_road_name_by_group(px, py, None)
        return best_name

    # ---------------- 交叉口小工具 ----------------
    def _get_or_create_intersection(self, pos: Vector, eps_cm: float = 1.0) -> Node:
        for nd in self.nodes:
            if getattr(nd, "type", "") != "intersection":
                continue
            if nd.position.distance(pos) <= eps_cm:
                return nd
        n = Node(Vector(pos.x, pos.y), "intersection")
        self.add_node(n)
        return n

    # =========================================================
    #                公共小工具（抽取出复用）
    # =========================================================
    def _nearest_node_at(self, xf: float, yf: float, tol_cm: float = 50.0) -> Optional[Node]:
        best, best_d2 = None, None
        tx, ty = float(xf), float(yf)
        for nd in self.nodes:
            dx, dy = tx - float(nd.position.x), ty - float(nd.position.y)
            d2 = dx*dx + dy*dy
            if best_d2 is None or d2 < best_d2 - 1e-9:
                best_d2 = d2; best = nd
        if best is not None and (best_d2 is not None) and best_d2 <= tol_cm*tol_cm:
            return best
        return None

    def _roles_at_xy(self, xf: float, yf: float, eps: float = 1.0) -> List[str]:
        roles = []
        for rec in getattr(self, "order_meta", []):
            pu, do = rec.get("pickup_node"), rec.get("dropoff_node")
            if pu and abs(xf - pu.position.x) <= eps and abs(yf - pu.position.y) <= eps:
                roles.append(f"pick up address of order {rec['id']}")
            if do and abs(xf - do.position.x) <= eps and abs(yf - do.position.y) <= eps:
                roles.append(f"drop off address of order {rec['id']}")
        return sorted(set(roles))

    def _display_name_of(self, node: Node) -> str:
        """优先用稳定编号后的显示名，否则用 type。"""
        name = getattr(node, "display_name", "").strip()
        if name:
            return name
        t = self._t_of(node) or "poi"
        return t

    def _road_name_for_node(self, nd: Node) -> str:
        """仅返回节点自带的 road_name（不做任何推断）。"""
        return getattr(nd, "road_name", "") or ""

    def _mk_item(self, kind: str, node: Node, dist_cm: float,
                 x_cm: Optional[float] = None, y_cm: Optional[float] = None,
                 force_type: Optional[str] = None, override_id: Optional[Any] = None) -> Dict[str, Any]:
        px = float(node.position.x) if x_cm is None else float(x_cm)
        py = float(node.position.y) if y_cm is None else float(y_cm)
        typ = force_type if force_type is not None else self._t_of(node)
        rn = self._road_name_for_node(node)
        item = {
            "kind": kind,                       # 'waypoint' / 'poi' / 'intersection'(for S)
            "id": str(node) if override_id is None else str(override_id),
            "type": typ,
            "name": self._display_name_of(node),
            "x": px, "y": py,
            "dist_cm": float(dist_cm),
            "is_dock": (node in self._dock_nodes),
            "roles": self._roles_at_xy(px, py),
            "label": "", "label_text": "",
            "road_name": rn,
            "_node": node,                      # 后续统一用最短路更新距离；特殊场景可覆盖
        }
        return item

    # =========================================================
    #                   1) 路网加载 + 骨架
    # =========================================================
    def import_roads(self, map_path: str):
        with open(map_path, 'r', encoding='utf-8') as f:
            roads_data = json.load(f)

        roads = roads_data['roads']
        for road in roads:
            # meters → cm
            start = Vector(road['start']['x'] * 100.0, road['start']['y'] * 100.0)
            end   = Vector(road['end']['x'] * 100.0, road['end']['y'] * 100.0)
            road_obj = Road(start, end)

            # 四个交叉口点（人行道两侧）
            normal_vector = Vector(road_obj.direction.y, -road_obj.direction.x)
            p1 = road_obj.start - normal_vector * Config.SIDEWALK_OFFSET + road_obj.direction * Config.SIDEWALK_OFFSET
            p2 = road_obj.end   - normal_vector * Config.SIDEWALK_OFFSET - road_obj.direction * Config.SIDEWALK_OFFSET
            p3 = road_obj.end   + normal_vector * Config.SIDEWALK_OFFSET - road_obj.direction * Config.SIDEWALK_OFFSET
            p4 = road_obj.start + normal_vector * Config.SIDEWALK_OFFSET + road_obj.direction * Config.SIDEWALK_OFFSET

            n1 = self._get_or_create_intersection(p1)
            n2 = self._get_or_create_intersection(p2)
            n3 = self._get_or_create_intersection(p3)
            n4 = self._get_or_create_intersection(p4)

            self._road_id += 1
            left_name  = f"{self._ordinal(self._road_id)} road (left)"
            right_name = f"{self._ordinal(self._road_id)} road (right)"

            for (a, b, nm) in ((n1, n2, left_name), (n3, n4, right_name)):
                if a is b:
                    continue
                self.add_edge(Edge(a, b))
                self._set_edge_meta(a, b, {"name": nm, "kind": "road"})

            for a, b in ((n1, n4), (n2, n3)):
                if a is b:
                    continue
                self.add_edge(Edge(a, b))
                self._set_edge_meta(a, b, {"name": "", "kind": "endcap"})

        self.connect_adjacent_roads()
        # 先建骨架，再插 waypoint，再给 waypoint 命名（用最近 skeleton 路名）
        self._build_skeleton()
        self.interpolate_waypoints(spacing_cm=5000)  # 每 50m 一个（只加 normal）
        self._assign_waypoint_road_names_from_skeleton()

    def connect_adjacent_roads(self, threshold_cm: Optional[float] = None):
        if threshold_cm is None:
            threshold_cm = Config.SIDEWALK_OFFSET * 2.0 + 100.0
        inters = [n for n in self.nodes if getattr(n, "type", "") == "intersection"]
        n = len(inters)
        for i in range(n):
            ni = inters[i]
            for j in range(i + 1, n):
                nj = inters[j]
                if self._connected(ni, nj):
                    continue
                if ni.position.distance(nj.position) <= threshold_cm:
                    if ni is nj:
                        continue
                    self.add_edge(Edge(ni, nj))
                    self._xwalk_id += 1
                    self._set_edge_meta(ni, nj, {"name": f"xwalk_{self._xwalk_id}", "kind": "crosswalk"})

    def interpolate_waypoints(self, spacing_cm: int = 1500):
        """
        在所有边上插入 normal 节点（waypoints）。
        注意：**不在这里**给 waypoint 赋 road_name；后续统一用骨架最近路名覆盖。
        """
        current_edges = list(self.edges)
        for edge in current_edges:
            a, b = edge.node1, edge.node2
            distance = getattr(edge, "weight", a.position.distance(b.position))
            num_points = int(distance // spacing_cm)
            if num_points <= 1:
                continue

            dx = b.position.x - a.position.x
            dy = b.position.y - a.position.y
            norm = (dx * dx + dy * dy) ** 0.5
            if norm <= 1e-9:
                continue
            ux, uy = dx / norm, dy / norm

            new_nodes = []
            for i in range(1, num_points):
                px = a.position.x + ux * (i * spacing_cm)
                py = a.position.y + uy * (i * spacing_cm)
                node = Node(Vector(px, py), "normal")
                self.add_node(node)
                new_nodes.append(node)

            old_meta = self._get_edge_meta(a, b)
            # 拆边
            if edge in self.edges:
                self.edges.remove(edge)
            while b in self.adjacency_list[a]:
                self.adjacency_list[a].remove(b)
            while a in self.adjacency_list[b]:
                self.adjacency_list[b].remove(a)
            if old_meta is not None:
                self.edge_meta.pop(self._edge_key(a, b), None)

            chain = [a] + new_nodes + [b]
            for u, v in zip(chain, chain[1:]):
                self.add_edge(Edge(u, v))
                if old_meta is not None:
                    self._set_edge_meta(u, v, old_meta)

    # ------ 元数据 ------
    @staticmethod
    def _ordinal(n: int) -> str:
        if 10 <= (n % 100) <= 20:
            suf = "th"
        else:
            suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suf}"

    # ------ 骨架构建（交叉口 + 合并边） ------
    def _build_skeleton(self, include_crosswalks: bool = True):
        inters = [nd for nd in self.nodes if getattr(nd, "type", "") == "intersection"]

        def kxy(nd: Node) -> Tuple[int, int]:
            return (int(round(nd.position.x)), int(round(nd.position.y)))

        key2canon: Dict[Tuple[int, int], Node] = {}
        for nd in inters:
            key = kxy(nd)
            if key not in key2canon:
                key2canon[key] = nd

        seen = set()
        edges_out: List[Tuple[Node, Node, str, str]] = []

        def walk(src: Node, nb: Node, kind: str, name: str) -> Optional[Tuple[Node, float]]:
            prev, cur = src, nb
            length = src.position.distance(nb.position)
            while True:
                if getattr(cur, "type", "") == "intersection":
                    return cur, float(length)
                nxt = None
                step = 0.0
                for v in self.adjacency_list.get(cur, []):
                    if v is prev:
                        continue
                    meta = self._get_edge_meta(cur, v)
                    if not meta:
                        continue
                    if meta.get("kind") != kind:
                        continue
                    if kind == "road" and (meta.get("name") or "") != name:
                        continue
                    nxt = v
                    step = cur.position.distance(v.position)
                    break
                if nxt is None:
                    return None
                length += step
                prev, cur = cur, nxt

        for s in inters:
            s_c = key2canon[kxy(s)]
            for nb in self.adjacency_list.get(s, []):
                meta = self._get_edge_meta(s, nb) or {}
                kind = meta.get("kind", "")
                name = (meta.get("name") or "") if kind in ("road", "crosswalk") else ""
                if kind == "road":
                    if not name:
                        continue
                elif kind == "crosswalk":
                    if not include_crosswalks:
                        continue
                else:
                    continue

                walked = walk(s, nb, kind, name)
                if not walked:
                    continue
                t, _ = walked
                t_c = key2canon[kxy(t)]
                if s_c is t_c:
                    continue
                key = (min(id(s_c), id(t_c)), max(id(s_c), id(t_c)), kind, name)
                if key in seen:
                    continue
                seen.add(key)
                edges_out.append((s_c, t_c, kind, name))

        self.graph_skel = Graph()
        skel_nodes_seen = set()
        for u, v, kind, name in edges_out:
            if u not in skel_nodes_seen:
                self.graph_skel.add_node(u); skel_nodes_seen.add(u)
            if v not in skel_nodes_seen:
                self.graph_skel.add_node(v); skel_nodes_seen.add(v)
            self.graph_skel.add_edge(Edge(u, v))
            self.graph_skel.set_edge_meta(u, v, {"kind": kind, "name": name or ""})

    # === waypoint 命名：用“最近骨架路名”（不看 h/v） ===
    def _assign_waypoint_road_names_from_skeleton(self):
        for nd in self.nodes:
            t = (getattr(nd, "type", "") or "").lower()
            if t != "normal":
                # intersection/door 等都不在此赋名；intersection 也避免歧义
                continue
            if nd in self._dock_nodes:
                continue
            # 最近 skeleton road（简单正确）
            best_name, best_d = "", None
            p = nd.position
            for e in self.graph_skel.edges:
                u, v = e.node1, e.node2
                meta = self.graph_skel.get_edge_meta(u, v) or {}
                if meta.get("kind") != "road":
                    continue
                name = meta.get("name") or ""
                if not name:
                    continue
                proj, _ = self._project_point_to_segment(p, u.position, v.position)
                d = p.distance(proj)
                if (best_d is None) or (d < best_d - 1e-9):
                    best_d, best_name = d, name
            nd.road_name = best_name or ""

    # === 辅助：把建筑连接“正交化”，并把两段边打上 aux_* ===
    def _ensure_orthogonal_dock(self, base_edge: Edge, proj: Vector, t: float) -> Node:
        """
        返回 dock_node：
        - 若 t∈[0,1]：在段内切边（原生 ensure_node_on_edge_at）
        - 若 t<0 或 t>1：新建延长点 P=proj，连：端点→P (aux_ext)，并返回 P
        """
        a, b = base_edge.node1, base_edge.node2
        if 0.0 <= t <= 1.0:
            nd = self.graph_full.ensure_node_on_edge_at(base_edge, proj, snap_cm=120.0)
            return nd

        # 选择更近端点
        end = a if t < 0.0 else b
        # 延长线上的点作为“dock”节点
        dock = Node(Vector(proj.x, proj.y), "normal")
        self.add_node(dock)
        if not self._connected(end, dock):
            self.add_edge(Edge(end, dock))
            self._set_edge_meta(end, dock, {"name": "", "kind": "aux_ext"})
        return dock

    # —— 计算“朝向为 yaw_q 的矩形外边中点”（door 落点）
    @staticmethod
    def _door_point_from_yaw(center: Vector, bbox_w: float, bbox_h: float, rect_yaw_deg: float, outward_yaw_deg: float) -> Vector:
        """
        给定矩形中心/宽高、矩形姿态 rect_yaw，以及“门外法线朝向” outward_yaw，
        返回门在矩形边界上的世界坐标（取外法线方向命中的边）。
        """
        half_w, half_h = 0.5 * bbox_w, 0.5 * bbox_h
        # 世界方向 → 转到 local
        d_world = Map._unit_from_angle(outward_yaw_deg)
        yaw_rad = math.radians(rect_yaw_deg)
        c, s = math.cos(-yaw_rad), math.sin(-yaw_rad)
        dx = d_world.x * c - d_world.y * s
        dy = d_world.x * s + d_world.y * c
        eps = 1e-9
        # 命中哪个边
        tx = (half_w / abs(dx)) if abs(dx) > eps else float("inf")
        ty = (half_h / abs(dy)) if abs(dy) > eps else float("inf")
        if tx < ty:
            # 命中左右边：x=±half_w
            x_local = math.copysign(half_w, dx)
            y_local = (dy / max(abs(dx), eps)) * x_local
        else:
            # 命中上下边：y=±half_h
            y_local = math.copysign(half_h, dy)
            x_local = (dx / max(abs(dy), eps)) * y_local
        # 回到世界
        c2, s2 = math.cos(yaw_rad), math.sin(yaw_rad)
        wx = x_local * c2 - y_local * s2
        wy = x_local * s2 + y_local * c2
        return Vector(center.x + wx, center.y + wy)

    # =========================================================
    #                   2) POI/建筑一次性建立（仅用 yaw）
    # =========================================================
    def _find_existing_node_at(self, pos: Vector, eps_cm: float = 80.0) -> Optional[Node]:
        return self.graph_full.find_existing_node_at(pos, eps_cm)

    def import_pois(self, world_json: str):
        """
        - 建筑类（restaurant/store/rest_area/hospital/car_rental/customer/building/BP_Building_*）
          以“门”的世界坐标作为吸附/取路名的参考点（近骨架；建筑仍保留 h/v 优先逻辑）。
        - 点状（charging_station/bus_station）
          road_name = 最近骨架路；自身与 dock 之间连 aux_perp

        - 稳定编号：为所有 POI（含 building）生成 display_name（如 "restaurant 2"）。
        """
        DOOR_YAW_OFFSET_DEG = 90.0  # 仅用于门位置修正

        with open(world_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        building_like = {"restaurant", "store", "rest_area", "hospital", "car_rental", "customer", "building"}
        point_like    = {"charging_station", "bus_station"}

        nodes = data.get("nodes", [])

        for obj in nodes:
            props = obj.get("properties", {}) or {}
            inst  = str(obj.get("instance_name", "") or "")
            loc   = props.get("location", {}) or {}
            ori   = props.get("orientation", {}) or {}
            bbox  = props.get("bbox", {}) or {}
            pt    = (props.get("poi_type") or props.get("type") or "").strip().lower()

            # 判定类型
            if pt in building_like:
                tname = pt
                is_building = True
            elif inst.startswith("BP_Building"):
                tname = "building"
                is_building = True
            elif pt in point_like:
                tname = pt
                is_building = False
            else:
                continue

            cx, cy = float(loc.get("x", 0.0)), float(loc.get("y", 0.0))
            center = Vector(cx, cy)

            # ======= 建筑：按门来算 =======
            if is_building:
                # 量化 yaw→分组(h/v)；注意：分组不加 90° 偏移
                yaw_deg_raw = float((ori.get("yaw", 0.0) or 0.0))
                yaw_q       = self._quantize_yaw_deg(yaw_deg_raw)
                group       = self._hv_group_from_yaw(yaw_q)

                # 先计算门的落点（仅门位置加 90° 修正）
                door_pos = None
                w = float(bbox.get("x", 0.0))
                h = float(bbox.get("y", 0.0))
                yaw_rect = float(ori.get("yaw", 0.0) or 0.0)
                if w > 0 and h > 0:
                    outward = (yaw_q + DOOR_YAW_OFFSET_DEG) % 360.0
                    door_pos = self._door_point_from_yaw(center, w, h, yaw_rect, outward)

                # 取路名/吸附的参考点：优先门，其次中心
                ax = (door_pos.x if door_pos else center.x)
                ay = (door_pos.y if door_pos else center.y)

                # 最近骨干路名（优先同组）
                road_name = self._nearest_skeleton_road_name_by_group(ax, ay, group)

                # POI 自身节点（建筑中心）
                poi_node = Node(center, type=tname)
                poi_node.road_name = road_name
                seq = self._poi_seq_counter.get(tname, 0) + 1
                self._poi_seq_counter[tname] = seq
                poi_node.display_name = f"{tname} {seq}"
                self.pois.append(poi_node)

                # 吸附最近真实边（优先同组 + 优先 road）
                best = self._snap_to_nearest_edge_with_group(ax, ay, group)
                dock_node = None
                if best is not None:
                    proj = best["proj"]; t = best.get("t", 0.0)
                    dock_node = self._ensure_orthogonal_dock(best["edge"], proj, t)
                    self._dock_nodes.add(dock_node)
                    dock_node.road_name = road_name

                # 建立 door 节点 + 垂线（aux_perp）
                door_node = None
                if dock_node is not None and door_pos is not None:
                    door_node = Node(door_pos, "door")
                    if door_node not in self.nodes:
                        self.add_node(door_node)
                    if not self._connected(dock_node, door_node):
                        self.add_edge(Edge(dock_node, door_node))
                        self._set_edge_meta(dock_node, door_node, {"name": "", "kind": "aux_perp"})
                    door_node.road_name = road_name

                if door_node is not None:
                    self._door2poi[door_node] = poi_node

                # 记录（给 viewer 灰框/订单等）
                building_box = {
                    "x": cx, "y": cy, "w": float(bbox.get("x", 0.0)), "h": float(bbox.get("y", 0.0)),
                    "yaw": float(ori.get("yaw", 0.0)), "poi_type": tname
                } if bbox else None

                self.poi_meta.append({
                    "node": poi_node,
                    "dock_node": dock_node,
                    "door_node": door_node,
                    "road_name": road_name,
                    "center": (cx, cy),
                    "building_box": building_box,
                })
                continue  # 建筑分支结束

            # ======= 点状 POI：充电/公交 =======
            # 最近骨干路名（不分组）
            road_name = self._nearest_skeleton_road_name_by_group(cx, cy, None)

            poi_node = Node(center, type=tname)
            poi_node.road_name = road_name
            seq = self._poi_seq_counter.get(tname, 0) + 1
            self._poi_seq_counter[tname] = seq
            poi_node.display_name = f"{tname} {seq}"
            self.pois.append(poi_node)

            # 吸附最近真实边（普通最近）
            best = self.snap_to_nearest_edge(cx, cy)
            dock_node = None
            if best is not None:
                proj = best["proj"]; t = best.get("t", 0.0)
                dock_node = self._ensure_orthogonal_dock(best["edge"], proj, t)
                self._dock_nodes.add(dock_node)
                dock_node.road_name = road_name

            # 自身 ←→ dock（aux_perp）
            if dock_node is not None:
                self.add_node(poi_node)
                if poi_node.position.distance(dock_node.position) > 1e-6:
                    self.add_edge(Edge(dock_node, poi_node))
                    self._set_edge_meta(dock_node, poi_node, {"name": "", "kind": "aux_perp"})

            # 记录
            self.poi_meta.append({
                "node": poi_node,
                "dock_node": dock_node,
                "door_node": None,
                "road_name": road_name,
                "center": (cx, cy),
                "building_box": None,
            })

    # =========================================================
    #                   3) 订单（只复用既有POI）
    # =========================================================
    def _pick_meta(self,tname: str, hint_xy: Optional[Tuple[float, float]]) -> Optional[Dict[str, Any]]:
        tname = (tname or "").strip().lower()
        cands = [m for m in self.poi_meta if self._t_of(m["node"]) == tname or (tname == "building" and self._t_of(m["node"]) == "building")]
        if not cands:
            return None
        if hint_xy is None:
            return cands[0]
        hx, hy = hint_xy
        return min(cands, key=lambda m: (m.get("center",(0,0))[0]-hx)**2 + (m.get("center",(0,0))[1]-hy)**2)
    
    def set_active_orders(self, orders: List[Dict[str, Any]], world_json_path: Optional[str] = None, eps_cm: float = 80.0) -> List[Dict[str, Any]]:
        """
        pickup/dropoff 只在既有 POI 集合里匹配（按类型 + hint 最近）。
        同时把命中的建筑灰框（若有）拷贝到 rec['*building']，保证 viewer 在 building-link 关闭时也能画出 dropoff 灰框。
        """
        self.order_meta = []

        def _xy_from_hint(h):
            if not isinstance(h, dict): return None
            return (float(h.get("x", 0.0)), float(h.get("y", 0.0)))

        results = []
        for od in orders:
            oid  = str(od.get("id",""))
            pt   = (od.get("pickup_type") or "restaurant")
            dt   = (od.get("dropoff_type") or "building")
            phxy = _xy_from_hint(od.get("pickup_hint"))
            dhxy = _xy_from_hint(od.get("dropoff_hint"))

            pm = self._pick_meta(pt, phxy)
            dm = self._pick_meta(dt, dhxy)

            rec = {
                "id": oid,
                "pickup_node": pm.get("door_node") or pm.get("dock_node") if pm else None,
                "dropoff_node": dm.get("door_node") or dm.get("dock_node") if dm else None,
                "pickup_building": (pm.get("building_box") if pm else None),
                "dropoff_building": (dm.get("building_box") if dm else None),
            }
            self.order_meta.append(rec)
            results.append(rec)
        return results

    def clear_active_orders(self):
        self.order_meta = []

    # =========================================================
    #                   4) Agent 包 + 可达 / POI 列表
    # =========================================================
    def _intersection_cluster(self, seed: Optional[Node]) -> set:
        if seed is None or self._t_of(seed) != "intersection":
            return set()
        vis = {seed}
        dq = deque([seed])
        while dq:
            x = dq.popleft()
            for nb in self.adjacency_list[x]:
                if self._t_of(nb) == "intersection" and nb not in vis:
                    vis.add(nb)
                    dq.append(nb)
        return vis

    def _adjacent_best_counting(
    self, seed: Node, agent_xy: Optional[Tuple[float, float]] = None
    ) -> Tuple[Optional[Node], Optional[float]]:
        """
        从 seed 出发，在“只经过非计数点”的条件下，找到相邻计数点（可能有多个）。
        - 若 agent_xy 为 None：保持原语义，找到第一个就立即返回 (neighbor, dist_from_seed)。
        - 若 agent_xy 提供：枚举所有相邻计数点，用最短路距离
        shortest_path_xy_to_node(agent_xy -> neighbor) 选离 agent 最近的那个，
        返回 (best_neighbor, dist_from_seed_of_that_neighbor)。
        找不到则返回 (None, None)。
        """
        def _is_seed_candidate(nd: Node) -> bool:
            t = (getattr(nd, "type", "") or "").lower()
            return t in ("normal", "intersection") and (nd not in self._dock_nodes)

        from collections import deque

        visited = {seed}
        q = deque()

        # 先把 seed 的非计数邻居入队；若直接邻居是计数点且 agent_xy=None，可直接早退
        for v in self.adjacency_list.get(seed, []):
            step = seed.position.distance(v.position)
            if _is_seed_candidate(v):
                if agent_xy is None:
                    return v, step
                else:
                    q.clear()  # 不早退，改为统一收集
            else:
                visited.add(v)
                q.append((v, step))

        # BFS 只在“非计数点”层上扩展，收集所有相邻计数点
        neighbors: List[Tuple[Node, float]] = []
        # 再检查一遍直接邻居（agent_xy!=None 的情况，需要把刚才的计数邻居也纳入列表）
        for v in self.adjacency_list.get(seed, []):
            step = seed.position.distance(v.position)
            if _is_seed_candidate(v):
                neighbors.append((v, step))

        while q:
            u, dacc = q.popleft()
            for v in self.adjacency_list.get(u, []):
                if v in visited:
                    continue
                step = u.position.distance(v.position)
                if _is_seed_candidate(v):
                    if agent_xy is None:
                        return v, dacc + step
                    neighbors.append((v, dacc + step))
                else:
                    visited.add(v)
                    q.append((v, dacc + step))

        if not neighbors:
            return None, None

        if agent_xy is None:
            # 找到了一批，但保持旧语义：返回第一个（通常就是最近层）
            return neighbors[0]

        ax, ay = agent_xy
        best = None
        best_total = None

        for nb, d_seed_nb in neighbors:
            # 用“agent → nb”的最短路距离进行选择（更稳妥）
            _path, total_cm, _ = self.shortest_path_xy_to_node(ax, ay, nb)
            if not math.isfinite(total_cm):
                continue
            if best is None or total_cm < best_total - 1e-6:
                best, best_total = (nb, d_seed_nb), total_cm

        return best if best is not None else (None, None)

    
    def get_reachable_set_xy(self, x: float, y: float, include_docks: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """
        距离统一用 shortest_path_xy_to_node(x,y,node) +（必要时）off-graph 直线段。
        当当前位置不在任何图节点附近时，会把 (x,y)→最近真实线段投影点 的直线距离
        作为固定前缀加入所有条目的 dist_cm，避免出现 0.0m 的假小值。
        """
        res = {"next_hop": [], "next_intersections": []}
        from collections import deque

        # ---------- helpers ----------
        def _t(nd): return (getattr(nd, "type", "") or "").lower()
        def _is_dock(nd): return nd in self._dock_nodes
        def _is_counting(nd):  # BFS 的“下一计数点”判定
            return _t(nd) in ("normal", "intersection") and (include_docks or not _is_dock(nd))
        def _is_seed_candidate(nd):  # 作为 waypoint 的候选（无条件排除 dock）
            return _t(nd) in ("normal", "intersection") and (not _is_dock(nd))

        # 当前位置是否在节点上
        node0 = self._nearest_node_at(x, y, tol_cm=50.0)

        # 计算 off-graph 直线距离（仅当不在节点上）
        off_graph_cm = 0.0
        if node0 is None:
            snap = self.snap_to_nearest_edge(x, y, use_segment=True)
            if snap is not None:
                px, py = float(snap["proj"].x), float(snap["proj"].y)
                off_graph_cm = float(Vector(x, y).distance(Vector(px, py)))

        # 距离缓存（把 off_graph_cm 加到所有目标）
        _dist_cache: Dict[int, float] = {}
        def _dist_from_agent(nd: Node) -> float:
            nid = id(nd)
            if nid in _dist_cache:
                return _dist_cache[nid]
            _path, dist_cm, _ = self.shortest_path_xy_to_node(x, y, nd)
            dist_total = float(dist_cm) + float(off_graph_cm)
            _dist_cache[nid] = dist_total
            return dist_total

        items_by_key: Dict[tuple, Dict[str, Any]] = {}
        def _put_item(it):
            key = ("id", it["id"]) if it.get("id") else ("xy", it["type"], int(round(it["x"])), int(round(it["y"])))
            if key not in items_by_key:
                items_by_key[key] = it

        # 从 start 沿“只在非计数点上扩展”的 BFS，收集所有相邻计数点（路径中不跨其他计数点）
        def _adjacent_countings(start):
            out: List[Node] = []
            visited = {start}
            q = deque()

            # 先考察直接邻居
            for v in self.adjacency_list.get(start, []):
                if _is_seed_candidate(v):
                    out.append(v)
                else:
                    visited.add(v)
                    q.append(v)

            # 继续在非计数点上扩展
            while q:
                u = q.popleft()
                for v in self.adjacency_list.get(u, []):
                    if v in visited:
                        continue
                    if _is_seed_candidate(v):
                        out.append(v)
                    else:
                        visited.add(v)
                        q.append(v)

            # 去重
            by_id: Dict[int, Node] = {}
            for nd in out:
                k = id(nd)
                if k not in by_id:
                    by_id[k] = nd
            return list(by_id.values())

        # 从 start（非计数点/POI）按 hop 找最近的 K 个计数点；计数点不继续扩展
        def _nearest_k_countings_by_hop(start, K=2):
            found: Dict[int, Tuple[Node, int]] = {}
            visited = set([start])
            q = deque([(start, 0)])
            while q and len(found) < K:
                u, hops = q.popleft()
                for v in self.adjacency_list.get(u, []):
                    if v in visited:
                        continue
                    if _is_seed_candidate(v):
                        vid = id(v)
                        if (vid not in found) or (hops+1 < found[vid][1]):
                            found[vid] = (v, hops+1)
                    else:
                        visited.add(v)
                        q.append((v, hops+1))
            arr = sorted(found.values(), key=lambda t: t[1])[:K]
            return [nd for (nd, _h) in arr]

        # BFS：从一个计数点 side 出发，直到遇到“下一个计数点”为止（不扩展该点）；途中收集 POI
        door2poi = getattr(self, "_door2poi", {}) or {}
        def _bfs_until_next_counting(side, node0_ref):
            q = deque([side])
            visited = {side}
            while q:
                u = q.popleft()

                # 碰到下一个计数点：不再扩展该点，但队列其它分支继续
                if (u is not side) and _is_counting(u):
                    continue

                ut = self._t_of(u)
                roles_here = self._roles_at_xy(float(u.position.x), float(u.position.y))
                parent = door2poi.get(u) if ut == "door" else None
                parent_type = (getattr(parent, "type", "") or "").lower() if parent else None

                # 只收 door/charging/bus（不把 dock 收进 N）
                is_poi = (u is not side and u is not node0_ref) and (
                    ut in ("charging_station", "bus_station") or
                    (ut == "door" and (parent_type != "building" or len(roles_here) != 0))
                )
                if is_poi:
                    if parent is not None:
                        it = self._mk_item(
                            "poi", parent, 0.0,
                            x_cm=float(u.position.x), y_cm=float(u.position.y),
                            force_type=self._t_of(parent), override_id=parent
                        )
                        # 距离要按“门”算
                        it["_node"] = u
                    else:
                        it = self._mk_item("poi", u, 0.0)
                    _put_item(it)

                # 扩展（POI 优先，使之更早被弹出记录）
                poi_first, others = [], []
                for v in self.adjacency_list.get(u, []):
                    if v in visited:
                        continue
                    vt = self._t_of(v)
                    (poi_first if vt in ("door", "charging_station", "bus_station") or (v in self._dock_nodes) else others).append(v)
                for v in poi_first + others:
                    visited.add(v)
                    q.append(v)

        # ---------- 起点模式分支 ----------
        seeds_waypoints: List[Node] = []
        seeds_bfs: List[Node] = []

        if node0 is not None:
            # 在图上的一个节点
            if _is_seed_candidate(node0):
                # 在计数点：不把 node0 自己进 N；相邻计数点都进 N；BFS 从【node0 + 相邻计数点】出发
                adj = _adjacent_countings(node0)  # 可能是 1~N 个
                for nd in adj:
                    if _is_seed_candidate(nd):
                        seeds_waypoints.append(nd)
                seeds_bfs = ([node0] + seeds_waypoints)
            else:
                # 在 POI/非计数点：按 hop 找最近两个计数点
                seeds_waypoints = _nearest_k_countings_by_hop(node0, K=2)
                seeds_bfs = list(seeds_waypoints)
        else:
            # 不在节点上：走“投影→两端→补成两个相邻计数点”逻辑
            snap = self.snap_to_nearest_edge(x, y, use_segment=True)
            if snap is None:
                return res
            edge, proj = snap["edge"], snap["proj"]
            a, b = edge.node1, edge.node2

            def _find_nearest_counting(start):
                if _is_seed_candidate(start):
                    return start
                q = deque([start]); visited = {start}
                while q:
                    u = q.popleft()
                    for v in self.adjacency_list.get(u, []):
                        if v in visited:
                            continue
                        if _is_seed_candidate(v):
                            return v
                        visited.add(v)
                        q.append(v)
                return None

            seedA = _find_nearest_counting(a)
            seedB = _find_nearest_counting(b)
            seeds = []
            if seedA is not None: seeds.append(seedA)
            if seedB is not None: seeds.append(seedB)

            if seeds:
                if len(seeds) == 1 or (len(seeds) >= 2 and seeds[0] is seeds[1]):
                    only_seed = seeds[0]
                    # 选“离 agent 最近”的相邻计数点
                    neighbor, _ = self._adjacent_best_counting(only_seed, agent_xy=(x, y))
                    if neighbor is not None:
                        seeds_waypoints = [only_seed, neighbor]
                    else:
                        seeds_waypoints = [only_seed]
                else:
                    seeds_waypoints = seeds[:2]
            seeds_bfs = list(seeds_waypoints)

        # ---------- waypoint 写入（不含“当前计数点自己”） ----------
        seen_ids = set()
        for nd in seeds_waypoints:
            if id(nd) in seen_ids:
                continue
            seen_ids.add(id(nd))
            if _is_seed_candidate(nd):  # 排除 dock
                _put_item(self._mk_item("waypoint", nd, 0.0))

        # ---------- BFS 收集 POI ----------
        node0_ref = node0
        for nd in seeds_bfs:
            _bfs_until_next_counting(nd, node0_ref)

        # ---------- 统一用最短路更新距离，并排序/打标签（Next Hops） ----------
        for it in items_by_key.values():
            nd = it.get("_node", None)
            if isinstance(nd, Node):
                it["dist_cm"] = float(_dist_from_agent(nd))
            else:
                nd2 = self._nearest_node_at(it["x"], it["y"], tol_cm=60.0)
                if nd2 is not None:
                    it["_node"] = nd2
                    it["dist_cm"] = float(_dist_from_agent(nd2))
                else:
                    it["dist_cm"] = float("inf")

        next_all = sorted(items_by_key.values(), key=lambda d: d["dist_cm"])
        for i, it in enumerate(next_all, 1):
            # waypoint 且其实是交叉口 → 文案写 intersection
            if it["kind"] == "waypoint" and it["type"] == "intersection":
                shown_name = "intersection"
            elif it["kind"] == "waypoint":
                shown_name = "waypoint"
            else:
                shown_name = it.get("name") or (it.get("type") or "poi")

            dist_m = round(float(it["dist_cm"]) / 100.0, 1)
            rn = it.get("road_name") or ""
            # intersection 不显示路名
            rn_tail = (f" • {rn}" if (rn and it.get("type") != "intersection") else "")
            roles = it.get("roles") or []
            roles_txt = (" / " + " / ".join(roles)) if roles else ""
            it["label"] = f"N{i}"
            it["label_text"] = f"N{i}: {shown_name}{roles_txt} at {_fmt_xy_m(it['x'], it['y'])} • {dist_m}m{rn_tail}"

        # =============== 计算 S（next_intersections） ===============
        # 工具：通过坐标找回节点
        def _node_by_xy(xx, yy, tol_cm=50.0):
            return self._nearest_node_at(xx, yy, tol_cm=tol_cm)

        # 从任意起点出发，穿过非 intersection，命中的“第一层 intersection”集合（命中即停，不从 intersection 扩展）
        def _first_intersections_from(start):
            hits = set()
            seen = {start}
            q = deque([start])
            while q:
                u = q.popleft()
                for v in self.adjacency_list.get(u, []):
                    if v in seen:
                        continue
                    meta = (self._get_edge_meta(u, v) or {})
                    kind = (meta.get("kind") or "")
                    if kind.startswith("aux_"):
                        continue
                    if self._t_of(v) == "intersection":
                        hits.add(v)     # 命中即收，不再从 v 扩展
                        continue
                    seen.add(v)
                    q.append(v)
            return hits

        # 1) 收集显式的 intersection 种子（自己是 intersection + next_hop 中的 intersection）
        seeds_S = set()
        if node0 is not None and self._t_of(node0) == "intersection":
            seeds_S.add(node0)
        for it in next_all:
            if it["kind"] == "waypoint" and it["type"] == "intersection":
                nd = _node_by_xy(it["x"], it["y"], tol_cm=60.0)
                if nd is not None:
                    seeds_S.add(nd)

        S_items: List[Dict[str, Any]] = []

        if seeds_S:
            # --- 有显式 intersection 种子：簇 → 前沿（不同簇的第一层 intersection） ---
            cluster = set()
            for s in seeds_S:
                cluster |= self._intersection_cluster(s)

            def _frontiers_from_cluster(cluster_set):
                front = set()
                visited = set(cluster_set)
                q = deque()
                # 从簇中每个 intersection 的邻居出发
                for u in list(cluster_set):
                    for v in self.adjacency_list.get(u, []):
                        meta = (self._get_edge_meta(u, v) or {})
                        kind = (meta.get("kind") or "")
                        if kind.startswith("aux_"):
                            continue
                        if v in visited:
                            continue
                        if self._t_of(v) == "intersection":
                            if v not in cluster_set:
                                front.add(v)
                            continue
                        visited.add(v)
                        q.append(v)
                # 继续穿过非 intersection 拓展
                while q:
                    u = q.popleft()
                    for v in self.adjacency_list.get(u, []):
                        if v in visited:
                            continue
                        meta = (self._get_edge_meta(u, v) or {})
                        kind = (meta.get("kind") or "")
                        if kind.startswith("aux_"):
                            continue
                        if self._t_of(v) == "intersection":
                            if v not in cluster_set:
                                front.add(v)
                            continue
                        visited.add(v)
                        q.append(v)
                return list(front)

            frontiers = _frontiers_from_cluster(cluster)

            # 组装 S_items（距离用最短路从 (x,y) 到该 intersection）+ off_graph_cm
            for nd in frontiers:
                dist_cm = float(_dist_from_agent(nd))
                item = self._mk_item("intersection", nd, dist_cm, force_type="intersection")
                S_items.append(item)

        else:
            # --- 无显式 intersection 种子：对每个“下一跳 waypoint(normal)”直接取其“第一层 intersection”作为 S ---
            cand_S = set()
            for it in next_all:
                if it["kind"] != "waypoint":
                    continue
                # 仅 normal 的“下一跳 intersection”，intersection 自己不在此分支
                if it["type"] == "intersection":
                    continue
                nd = _node_by_xy(it["x"], it["y"], tol_cm=60.0)
                if nd is None:
                    continue
                cand_S |= _first_intersections_from(nd)

            for nd in cand_S:
                dist_cm = float(_dist_from_agent(nd))
                item = self._mk_item("intersection", nd, dist_cm, force_type="intersection")
                S_items.append(item)

        # 排序 & 打 S 标签（intersection 不显示路名）
        S_items.sort(key=lambda d: d["dist_cm"])
        for i, it in enumerate(S_items, 1):
            dist_m = round(float(it["dist_cm"]) / 100.0, 1)
            it["label"] = f"S{i}"
            it["label_text"] = f"S{i}: intersection at {_fmt_xy_m(it['x'], it['y'])} • {dist_m}m"

        # 输出
        res["next_hop"] = next_all
        res["next_intersections"] = S_items
        return res



    # -------- POIs（最短路距离排序） --------
    def shortest_path_xy_to_node(self, x: float, y: float, target: Node):
        return self.graph_full.shortest_path_xy_to_node(x, y, target)

    def shortest_path_nodes(self, start: Node, target: Node):
        return self.graph_full.shortest_path_nodes(start, target)

    def list_direct_reachable_pois_xy(self, x: float, y: float) -> List[Dict[str, Any]]:
        # building 也纳入（用于 dropoff 展示），anchor 用 door/dock
        building_like = {"restaurant", "store", "rest_area", "hospital", "car_rental", "customer", "building"}
        out = []
        for meta in self.poi_meta:
            pnode = meta["node"]
            ptype = self._t_of(pnode)
            if ptype in building_like:
                anchor = meta.get("door_node") or meta.get("dock_node")
            else:
                anchor = pnode or meta.get("dock_node")
            if anchor is None:
                continue
            _path, dist_cm, _ = self.shortest_path_xy_to_node(x, y, anchor)
            if not math.isfinite(dist_cm):
                continue
            out.append({
                "id": str(pnode),
                "type": ptype,
                "name": self._display_name_of(pnode),
                "x": anchor.position.x, "y": anchor.position.y,
                "dist_cm": float(dist_cm),
                "road_name": self._road_name_for_node(anchor),
            })
        out.sort(key=lambda d: d["dist_cm"])
        return out
    
    def agent_info_package_xy(self, x: float, y: float, include_docks: bool = False,
                              limit_next: int = 50, limit_s: int = 50, limit_poi: int = 200) -> Dict[str, Any]:
        reachable = self.get_reachable_set_xy(x, y, include_docks=include_docks)
        next_hop = list(reachable.get("next_hop", []))[:limit_next]
        next_intersections = list(reachable.get("next_intersections", []))[:limit_s]

        # 检测“当前位置是否在某个 POI 上”
        poi_here: Optional[str] = None
        road_here: str = ""
        roles_here: List[str] = []
        ref = Vector(x, y)

        best = None
        for meta in self.poi_meta:
            door = meta.get("door_node")
            pnode = meta.get("node")
            if door is not None:
                d = ref.distance(door.position)
                if d <= self.NEAR_EPS:
                    cand = (0, d, "door", door, meta)
                    if best is None or (cand[0], cand[1]) < (best[0], best[1]): best = cand
            if pnode is not None:
                t = (getattr(pnode, "type", "") or "").lower()
                if t in ("bus_station", "charging_station"):
                    d = ref.distance(pnode.position)
                    if d <= self.NEAR_EPS:
                        cand = (1, d, t, pnode, meta)
                        if best is None or (cand[0], cand[1]) < (best[0], best[1]): best = cand

        if best is not None:
            _, _, kind, nd, meta = best
            if kind == "door":
                poi_here = self._display_name_of(meta.get("node"))
            else:
                poi_here = self._display_name_of(nd)
            road_here = self._road_name_for_node(nd) or (meta.get("road_name") or "")
            roles_here = self._roles_at_xy(float(nd.position.x), float(nd.position.y), eps=1.0)

        # 全量 POI（含 building；稍后按是否订单相关过滤/排序）
        poi_list = self.list_direct_reachable_pois_xy(x, y)[:limit_poi]

        # 订单角色标签（保持）
        poi_xy2tags: Dict[tuple, List[str]] = {}
        for rec in getattr(self, "order_meta", []):
            oid = str(rec.get("id", "")); pu = rec.get("pickup_node"); do = rec.get("dropoff_node")
            if pu is not None:
                k = (int(round(pu.position.x)), int(round(pu.position.y)))
                poi_xy2tags.setdefault(k, []).append(f"pick up address of order {oid}")
            if do is not None:
                k = (int(round(do.position.x)), int(round(do.position.y)))
                poi_xy2tags.setdefault(k, []).append(f"drop off address of order {oid}")

        # 出文本
        def m(cm): return f"{round(float(cm) / 100.0, 1)}m"  # 距离仍保留 1 位小数
        def fmt_xy(xx, yy): return _fmt_xy_m(xx, yy, decimals=2)

        lines: List[str] = []
        pos_line = f"Agent position: {fmt_xy(x, y)}"
        if poi_here:
            roles_tail = (" • " + " / ".join(roles_here)) if roles_here else ""
            pos_line += f" • {poi_here}{roles_tail}" + (f" • {road_here}" if road_here else "")
        lines.append(pos_line)

        lines.append("\nNext hops:")
        for it in next_hop:
            # waypoint 且其实是交叉口 → 文案写 intersection
            if it.get("kind") == "waypoint" and it.get("type") == "intersection":
                shown_name = "intersection"
            elif it.get("kind") == "waypoint":
                shown_name = "waypoint"
            else:
                shown_name = it.get("name") or (it.get("type") or "poi")
            rn = it.get("road_name") or ""
            # intersection 不显示路名
            rn_tail = f" • {rn}" if (rn and it.get("type") != "intersection") else ""
            roles = it.get("roles") or []
            roles_txt = (" / " + " / ".join(roles)) if roles else ""
            lines.append(f"{it.get('label','N?')}: {shown_name}{roles_txt} at {fmt_xy(it.get('x',0), it.get('y',0))} • {m(it.get('dist_cm',0))}{rn_tail}")

        lines.append("\nNext intersections:")
        for it in next_intersections:
            # intersection 不显示路名
            lines.append(f"{it.get('label','S?')}: intersection at {fmt_xy(it.get('x',0), it.get('y',0))} • {m(it.get('dist_cm',0))}")

        # All POIs：**纯距离**排序（仍然过滤掉不是订单端点的 building）
        enriched_pois = []
        for p in poi_list:
            k = (int(round(p.get("x", 0))), int(round(p.get("y", 0))))
            role_tags = sorted(set(poi_xy2tags.get(k, [])))
            has_order_role = len(role_tags) > 0
            enriched_pois.append({
                **p,
                "roles": role_tags,
                "has_order_role": has_order_role,
            })

        enriched_pois_for_print = [p for p in enriched_pois if not (p.get("type") == "building" and not p.get("has_order_role"))]
        enriched_pois_for_print.sort(key=lambda d: d["dist_cm"])

        lines.append("\nAll POIs by shortest-path distance:")
        for p in enriched_pois_for_print:
            name = p.get("name") or (p.get("type") or "poi")
            rn = p.get("road_name") or ""
            roles = p.get("roles") or []
            title = f"{name}" if not roles else f"{name} / {' / '.join(roles)}"
            road_tail = f" {rn}" if rn else ""
            lines.append(f"{title}: at {fmt_xy(p.get('x', 0), p.get('y', 0))} • {m(p.get('dist_cm', 0))}{road_tail}")

        # 订单端点
        orders_out = []
        for rec in getattr(self, "order_meta", []):
            oid = str(rec.get("id", ""))
            for kind_key, tag in (("pickup_node", "pickup"), ("dropoff_node", "dropoff")):
                nd = rec.get(kind_key)
                if nd is None: continue
                _path, dist_cm, _ = self.shortest_path_xy_to_node(x, y, nd)
                if not math.isfinite(dist_cm): continue
                orders_out.append({
                    "order_id": oid, "kind": tag,
                    "x": float(nd.position.x), "y": float(nd.position.y),
                    "dist_cm": float(dist_cm),
                    "road_name": self._road_name_for_node(nd),
                })
        lines.append("\nOrder endpoints (shortest-path):")
        for o in orders_out:
            tag = "pick up" if o["kind"] == "pickup" else "drop off"
            lines.append(f"{tag} of order {o['order_id']}: at {fmt_xy(o['x'], o['y'])} • {m(o['dist_cm'])} {o['road_name']}")

        return {
            "agent_xy": {"x": float(x), "y": float(y)},
            "reachable": reachable,
            "pois": enriched_pois,   # 完整列表（含 building），供上层需要时再过滤
            "orders": orders_out,
            "text": "\n".join(lines),
        }


    # =========================================================
    #                   5) 导出/可视化辅助
    # =========================================================
    def export_agent_graph(self, include_crosswalks: bool = True, print_preview: bool = True) -> Dict[str, Any]:
        nodes_out = [{"id": str(n), "type": "intersection",
                      "x": float(n.position.x), "y": float(n.position.y),
                      "roles": []} for n in self.graph_skel.nodes]

        edges_out = []
        for e in self.graph_skel.edges:
            u, v = e.node1, e.node2
            meta = self.graph_skel.get_edge_meta(u, v) or {}
            kind = meta.get("kind", "")
            if kind == "crosswalk" and not include_crosswalks:
                continue
            name = meta.get("name", "")
            edges_out.append({
                "u": str(u), "v": str(v),
                "dist_cm": float(u.position.distance(v.position)),
                "kind": kind, "name": (name or ""),
            })
        if print_preview:
            print("Nodes:", len(nodes_out))
            print("Edges:", len(edges_out))
        return {"nodes": nodes_out, "edges": edges_out}

    def get_building_link_segments(self) -> List[Tuple[Tuple[float, float], Tuple[float, float], str]]:
        """返回所有 aux_* 连线段 [( (x1,y1),(x2,y2), kind ), ...]"""
        segs = []
        seen = set()
        for a in self.nodes:
            for b in self.adjacency_list.get(a, []):
                key = self._edge_key(a, b)
                if key in seen: continue
                seen.add(key)
                meta = self._get_edge_meta(a, b) or {}
                kind = (meta.get("kind") or "")
                if kind.startswith("aux_"):
                    segs.append(((float(a.position.x), float(a.position.y)),
                                 (float(b.position.x), float(b.position.y)), kind))
        return segs

    # ---------------- 任意点到任意点路径（在拷贝图上改动） ----------------
    def route_xy_to_xy(self, ax: float, ay: float, tx: float, ty: float, snap_cm: float = 120.0):
        if not self.graph_full.edges:
            return [(float(ax), float(ay)), (float(tx), float(ty))]

        g = self.graph_full.copy()

        snapA = g.snap_to_nearest_edge(ax, ay)
        snapB = g.snap_to_nearest_edge(tx, ty)
        if snapA is None or snapB is None:
            return [(float(ax), float(ay)), (float(tx), float(ty))]

        nA = g.ensure_node_on_edge_at(snapA["edge"], snapA["proj"], snap_cm=snap_cm)
        nB = g.ensure_node_on_edge_at(snapB["edge"], snapB["proj"], snap_cm=snap_cm)

        def _anchor(snap, n):
            if n is snap["a"] or n is snap["b"]:
                return float(n.position.x), float(n.position.y)
            return float(snap["proj"].x), float(snap["proj"].y)

        ancA = _anchor(snapA, nA)
        ancB = _anchor(snapB, nB)
        path_nodes, _ = g.shortest_path_nodes(nA, nB)

        route = [(float(ax), float(ay))]
        if route[-1] != ancA:
            route.append(ancA)
        for nd in path_nodes:
            pt = (float(nd.position.x), float(nd.position.y))
            if route[-1] != pt:
                route.append(pt)
        if route[-1] != ancB:
            route.append(ancB)
        end_pt = (float(tx), float(ty))
        if route[-1] != end_pt:
            route.append(end_pt)

        # 去重
        cleaned = []
        for xy in route:
            if not cleaned or cleaned[-1] != xy:
                cleaned.append(xy)
        return cleaned
