# -*- coding: utf-8 -*-

import time
import random
from dataclasses import dataclass, field
from threading import Lock
from typing import List, Optional, Dict, Any, Tuple

from simworld.utils.vector import Vector
from Base.Timer import VirtualClock

# ==================== 可调参数（仅距离计价） ====================

PAY_PER_KM: float = 10.0         # 每公里单价（不含取货费）
AVG_SPEED_MPS: float = 1.35      # ETA 兜底（m/s）
TIME_MULTIPLIER: float = 1.25    # 截止时限 = ETA * 该系数 * 随机扰动
EARNING_JITTER: Tuple[float, float] = (0.75, 1.35)  # 收益扰动
TIME_JITTER: Tuple[float, float] = (0.85, 1.25)     # 时限扰动


# ============================== FoodItem ==============================

@dataclass
class FoodItem:
    name: str
    category: str = ""
    odor: str = ""                       # 原 JSON: "odor"
    motion_sensitive: bool = False
    nonthermal_time_sensitive: bool = False
    prep_time_s: int = 0

    # --- 热学字段（用于袋内演化） ---
    serving_temp_c: float = 60.0         # 上餐温度（热食示例；冷饮/冷食在 JSON 里填 4~8°C；冷冻约 -2°C）
    safe_min_c: float = 50.0             # 口感/安全区间下界（热食示例）
    safe_max_c: float = 70.0             # 口感/安全区间上界（热食示例）
    heat_capacity: float = 1.0           # 相对热容（权重）

    # 运行时（虚拟时间相关）
    temp_c: float = 0.0                  # 真实温度；在“备餐完成/取餐瞬间”置为 serving_temp_c
    prepared_at_sim: float = 0.0         # = order.prep_ready_sim
    picked_at_sim: float = 0.0
    delivered_at_sim: float = 0.0

    def __str__(self) -> str:
        return self.name


# ============================== Order ==============================

@dataclass
class Order:
    """
    - 初始化时即绑定最近 door_node（Map._pick_meta），并读取 road_name
    - 路径与距离：Map.shortest_path_nodes(start_node, end_node) 返回 (path, dist_cm)
    - 仅按距离计价；ETA 用距离/均速估计
    - 备餐时间线（接单时由 DeliveryMan 写入 prep_started_sim / prep_ready_sim / prep_longest_s）
    """
    city_map: Any
    pickup_address: Vector
    delivery_address: Vector
    items: List[FoodItem] = field(default_factory=list)
    special_note: str = ""

    # —— 路由/计价结果 —— 
    path_nodes: List[Any] = field(default_factory=list)  # List[Node]
    distance_cm: float = 0.0
    eta_s: float = 0.0
    time_limit_s: float = 0.0
    earnings: float = 0.0

    # —— 已绑定的节点/道路名 —— 
    pickup_node: Any = None
    dropoff_node: Any = None
    pickup_road_name: str = ""
    dropoff_road_name: str = ""
    road_name: str = ""  # 聚合展示名（pickup → dropoff）

    # —— 状态与流程字段 —— 
    id: int = field(init=False, default=-1)
    is_accepted: bool = False
    has_picked_up: bool = False
    has_delivered: bool = False
    start_time: float = field(init=False, default_factory=time.time)
    delivery_time: float = 0.0
    delivery_men: List[Any] = field(default_factory=list)

    is_shared: bool = False
    meeting_point: Optional[Vector] = None

    _lock: Lock = field(init=False, default_factory=Lock)
    _id_counter: int = 0

    # --- preparation timeline (虚拟时间；接单时写入) ---
    prep_started_sim: Optional[float] = None
    prep_ready_sim: Optional[float] = None
    prep_longest_s: float = 0.0

    # --- 用于常规 prompt/提示 ---
    sim_started_s: float = 0.0
    sim_elapsed_active_s: float = 0.0
    sim_delivered_s: Optional[float] = None

    def __post_init__(self):
        # 分配 id
        self.id = Order._id_counter
        Order._id_counter += 1
        self._bind_nodes_initial()

    # ---------------- 内部：初始化时绑定最近节点 + 路名 ----------------
    def _bind_nodes_initial(self):
        sx, sy = float(self.pickup_address.x), float(self.pickup_address.y)
        tx, ty = float(self.delivery_address.x), float(self.delivery_address.y)

        if not hasattr(self.city_map, "_pick_meta"):
            raise AttributeError("Map 未实现 _pick_meta(kind, (x,y))")

        pu_meta = self.city_map._pick_meta("restaurant", (sx, sy))
        do_meta = self.city_map._pick_meta("building", (tx, ty))

        self.pickup_node = pu_meta.get("door_node")
        self.dropoff_node = do_meta.get("door_node")
        self.pickup_road_name = str(pu_meta.get("road_name", "") or "")
        self.dropoff_road_name = str(do_meta.get("road_name", "") or "")

        if self.pickup_node is None or self.dropoff_node is None:
            raise Exception(False)

        self.road_name = (
            self.pickup_road_name
            if self.pickup_road_name == self.dropoff_road_name
            else f"{self.pickup_road_name} -> {self.dropoff_road_name}"
        )

    # ---------------- 路由（严格 Map 接口） ----------------
    def plan_route(self) -> List[Any]:
        if not hasattr(self.city_map, "shortest_path_nodes"):
            raise AttributeError("Map 未实现 shortest_path_nodes(start, target)")

        res = self.city_map.shortest_path_nodes(self.pickup_node, self.dropoff_node)
        if not (isinstance(res, tuple) and len(res) == 2):
            raise TypeError("shortest_path_nodes 必须返回 (path_nodes, dist_cm)")
        path, dist_cm = res

        if not isinstance(path, list) or not isinstance(dist_cm, (int, float)):
            raise TypeError("shortest_path_nodes 返回类型应为 (List[Node], float)")
        if not path or dist_cm <= 0:
            raise Exception(False)

        self.path_nodes = path
        self.distance_cm = float(dist_cm)
        return self.path_nodes

    # ---------------- 计价与时限 ----------------
    def price_and_deadline(self):
        dist_m = self.distance_cm / 100.0
        self.eta_s = (dist_m / AVG_SPEED_MPS) if dist_m > 0 else 0.0
        dist_km = dist_m / 1000.0

        base = PAY_PER_KM * dist_km
        self.earnings = base * random.uniform(*EARNING_JITTER)
        self.time_limit_s = self.eta_s * TIME_MULTIPLIER * random.uniform(*TIME_JITTER)

    # ---------------- 一键计算（路由 + 计价） ----------------
    def compute(self) -> List[Any]:
        self.plan_route()
        self.price_and_deadline()
        return self.path_nodes

    def compute_with_map(self, city_map: Any = None, tol_cm: float = 50.0) -> List[Any]:
        return self.compute()

    # ---------------- 文本输出（单个订单） ----------------
    def _fmt_time(self, seconds: float) -> str:
        minutes = int((seconds + 59) // 60)  # ceil 向上取整
        return f"{minutes} min"

    def to_text(self) -> str:
        dist_m = self.distance_cm / 100.0
        dist_km = dist_m / 1000.0
        lines = [
            f"[Order #{self.id}]",
            f"  Pickup : ({self.pickup_address.x:.1f}, {self.pickup_address.y:.1f})  | road: {self.pickup_road_name}",
            f"  Dropoff: ({self.delivery_address.x:.1f}, {self.delivery_address.y:.1f}) | road: {self.dropoff_road_name}",
            f"  Path   : {len(self.path_nodes)} nodes",
            f"  Dist   : {dist_m:.1f} m  ({dist_km:.3f} km)",
            f"  ETA    : {self._fmt_time(self.eta_s)}",
            f"  Limit  : {self._fmt_time(self.time_limit_s)}",
            f"  $$     : ${self.earnings:.2f}",
        ]
        if self.items:
            # 展示每个 item 的目标温度与热容
            items_str = ", ".join([f"{it.name} ({it.serving_temp_c:.0f}°C)" for it in self.items])
            lines.append("  Items  : " + items_str)
        if self.special_note:
            lines.append(f"  Note   : {self.special_note}")

        # 备餐提示（若已有时间线）
        if self.prep_started_sim is None:
            lines.append("  Prep   : starts after accept")
        elif self.prep_ready_sim is not None:
            now_sim = self.sim_started_s + self.sim_elapsed_active_s
            remain = self.remaining_prep_s(now_sim)
            lines.append(f"  Prep   : {'ready' if remain <= 0 else f'ready in ~{self._fmt_time(remain)} (virtual)'}")


        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text()

    # ---------------- 给 Viewer 的简要 hint ----------------
    def to_active_order_hint(self) -> Dict[str, Any]:
        px, py = float(self.pickup_address.x), float(self.pickup_address.y)
        dx, dy = float(self.delivery_address.x), float(self.delivery_address.y)
        return dict(
            id=self.id,
            pickup_xy=(px, py),
            dropoff_xy=(dx, dy),
            distance_cm=self.distance_cm,
            eta_s=self.eta_s,
            earnings=self.earnings,
            road_name=self.road_name,
        )

    # ---------------- 备餐时间线 ----------------
    def is_ready_for_pickup(self, now_sim: float) -> bool:
        if self.prep_started_sim is None or self.prep_ready_sim is None:
            return False
        return float(now_sim) >= float(self.prep_ready_sim)

    def remaining_prep_s(self, now_sim: float) -> float:
        if self.prep_started_sim is None or self.prep_ready_sim is None:
            return float(self.prep_longest_s or 0.0)  # 没开始就认为还差最长时间
        r = float(self.prep_ready_sim) - float(now_sim)
        return r if r > 0.0 else 0.0


# =========================== OrderManager ===========================

class OrderManager:
    """
    固定容量订单池：
    - 只从 world_nodes 中随机抽样（pickup=restaurant；dropoff=building）
    - 只接受 JSON 字典列表菜单（menu = data["items"]）
    - 生成订单时随机分配 1~4 个菜品（每个菜品按 JSON 字段构建 FoodItem）
    """

    def __init__(self, capacity: int = 10, menu: Optional[List[Dict[str, Any]]] = None, clock: Optional[VirtualClock] = None):
        assert capacity > 0
        self.capacity = int(capacity)
        self._orders: List[Order] = []
        self._menu: List[Dict[str, Any]] = list(menu) if menu else []
        self._clock = clock if clock is not None else VirtualClock()

    # ---------- 基础访问 ----------
    def __len__(self): return len(self._orders)
    def __iter__(self): return iter(self._orders)
    def list_orders(self) -> List[Order]: return list(self._orders)

    def get(self, order_id: int) -> Optional[Order]:
        for o in self._orders:
            if o.id == order_id:
                return o
        return None

    # ---------- 世界节点筛选 ----------
    @staticmethod
    def _is_restaurant(node: Dict[str, Any]) -> bool:
        props = node.get("properties", {})
        poi = str(props.get("poi_type") or props.get("type") or "").lower()
        return poi == "restaurant"

    @staticmethod
    def _is_building(node: Dict[str, Any]) -> bool:
        if str(node.get("instance_name", "")).startswith("BP_Building"):
            return True
        props = node.get("properties", {})
        t = str(props.get("type") or "").lower()
        return t == "building"

    @staticmethod
    def _xy_from_props(node: Dict[str, Any]) -> Tuple[float, float]:
        props = node.get("properties", {})
        loc = props.get("location", {})
        return float(loc.get("x", 0.0)), float(loc.get("y", 0.0))

    @staticmethod
    def _collect_candidates(world_nodes: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not world_nodes:
            raise Exception(False)
        pickups = [n for n in world_nodes if OrderManager._is_restaurant(n)]
        dropoffs = [n for n in world_nodes if OrderManager._is_building(n)]
        if not pickups or not dropoffs:
            raise Exception(False)
        return pickups, dropoffs

    # ---------- 生成 order.items（严格 JSON 字段） ----------
    def _spawn_items(self, min_n: int = 1, max_n: int = 4) -> List[FoodItem]:
        """
        从 self._menu（JSON 字典列表）随机挑选 1~4 个条目，生成 FoodItem。
        缺失字段按品类默认值补齐；temp_c 保持 0.0（备餐完成/取餐时由上层置为 serving_temp_c）。
        """
        if not isinstance(self._menu, list) or not self._menu:
            # 兜底：没有菜单时也给几样
            self._menu = [
                {"name": "Burger", "category": "HOT"},
                {"name": "Fries", "category": "HOT"},
                {"name": "Milk Tea", "category": "HOT"},
                {"name": "Salad", "category": "COLD"},
                {"name": "IceCream", "category": "FROZEN"},
            ]

        CAT_DEFAULTS = {
            "HOT":     {"serving_temp_c": 60.0, "safe_min_c": 50.0, "safe_max_c": 70.0, "heat_capacity": 1.0},
            "COLD":    {"serving_temp_c":  8.0, "safe_min_c":  2.0, "safe_max_c": 12.0, "heat_capacity": 0.9},
            "FROZEN":  {"serving_temp_c": -2.0, "safe_min_c": -5.0, "safe_max_c":  2.0, "heat_capacity": 0.95},
            "AMBIENT": {"serving_temp_c": 23.0, "safe_min_c": 15.0, "safe_max_c": 30.0, "heat_capacity": 0.85},
        }

        n = random.randint(min_n, max_n)
        picks = random.sample(self._menu, k=min(n, len(self._menu)))

        items: List[FoodItem] = []
        for t in picks:
            if not isinstance(t, dict):
                raise TypeError("OrderManager._spawn_items 期望 self._menu 为 JSON 字典列表")

            name = str(t.get("name", "Item"))
            category = str(t.get("category", "AMBIENT")).upper() or "AMBIENT"
            dflt = CAT_DEFAULTS.get(category, CAT_DEFAULTS["AMBIENT"])

            serving_temp_c = float(t["serving_temp_c"]) if "serving_temp_c" in t else float(dflt["serving_temp_c"])
            safe_min_c     = float(t["safe_min_c"])     if "safe_min_c"     in t else float(dflt["safe_min_c"])
            safe_max_c     = float(t["safe_max_c"])     if "safe_max_c"     in t else float(dflt["safe_max_c"])
            odor                        = str(t.get("odor", ""))
            motion_sensitive            = bool(t.get("motion_sensitive", False))
            nonthermal_time_sensitive   = bool(t.get("nonthermal_time_sensitive", False))
            prep_time_s                 = int(t.get("prep_time_s", 0))
            heat_capacity               = float(t.get("heat_capacity", dflt["heat_capacity"]))

            items.append(FoodItem(
                name=name,
                category=category,
                odor=odor,
                motion_sensitive=motion_sensitive,
                nonthermal_time_sensitive=nonthermal_time_sensitive,
                prep_time_s=prep_time_s,
                serving_temp_c=serving_temp_c,
                safe_min_c=safe_min_c,
                safe_max_c=safe_max_c,
                heat_capacity=heat_capacity,
                temp_c=float("nan"),  # 备餐完成/取餐时再置为 serving_temp_c
            ))

        return items

    # ---------- 生成单个订单 ----------
    def _spawn_one_order(self, city_map: Any, world_nodes: List[Dict[str, Any]]) -> Order:
        pickups, dropoffs = self._collect_candidates(world_nodes)
        pu_node = random.choice(pickups)
        do_node = random.choice(dropoffs)
        px, py = self._xy_from_props(pu_node)
        dx, dy = self._xy_from_props(do_node)

        items = self._spawn_items(1, 4)
        order = Order(
            city_map=city_map,
            pickup_address=Vector(px, py),
            delivery_address=Vector(dx, dy),
            items=items
        )
        order.compute()

        longest = 0
        if items:
            longest = max(int(getattr(it, "prep_time_s", 0) or 0) for it in items)
        order.prep_longest_s = float(longest)

        return order


    # ---------- 填充到固定容量 ----------
    def fill_pool(self, city_map: Any, world_nodes: List[Dict[str, Any]]):
        while len(self._orders) < self.capacity:
            self._orders.append(self._spawn_one_order(city_map, world_nodes))

    # ---------- 维护池：完成/移除并补齐 ----------
    def complete_order(self, order_id: int, city_map: Any, world_nodes: List[Dict[str, Any]]):
        target = self.get(order_id)
        if target is None:
            return
        target.has_delivered = True
        self._orders = [o for o in self._orders if o.id != order_id]
        self.fill_pool(city_map, world_nodes)

    def remove_order(self, order_id: int, city_map: Any, world_nodes: List[Dict[str, Any]]):
        self._orders = [o for o in self._orders if o.id != order_id]
        self.fill_pool(city_map, world_nodes)

    def accept_order(self, order_id: int) -> bool:
        o = self.get(order_id)
        if o is None:
            return False
        o.is_accepted = True
        return True

    def recompute_all(self):
        for o in self._orders:
            o.compute()

    # ---------- 一键文本输出 ----------
    def orders_text(self) -> str:
        if not self._orders:
            return "No orders."
        lines: List[str] = []
        lines.append(f"Orders ({len(self._orders)}):")
        for o in self._orders:
            lines.append(o.to_text())
            lines.append("-" * 60)
        return "\n".join(lines)

    # ---------- 输出给 MapViewer（兼容） ----------
    def to_active_order_hints(self) -> List[Dict[str, Any]]:
        return [
            {
                **o.to_active_order_hint(),
                "pickup_road": o.pickup_road_name,
                "dropoff_road": o.dropoff_road_name,
            }
            for o in self._orders
        ]

    def status_text(self) -> str:
        return "Active"
