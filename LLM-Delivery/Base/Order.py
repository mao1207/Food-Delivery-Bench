# -*- coding: utf-8 -*-

import time
import random
import math
from dataclasses import dataclass, field
from threading import Lock, RLock
from typing import List, Optional, Dict, Any, Tuple, Iterable, Union

from simworld.utils.vector import Vector
from Base.Timer import VirtualClock

# ==================== 可调参数（仅距离计价） ====================

PAY_PER_KM: float = 10.0         # 每公里单价（不含取货费）
AVG_SPEED_MPS: float = 1.35      # ETA 兜底（m/s）
TIME_MULTIPLIER: float = 1.25    # 截止时限 = ETA * 该系数 * 随机扰动
EARNING_JITTER: Tuple[float, float] = (0.75, 1.35)  # 收益扰动
TIME_JITTER: Tuple[float, float] = (0.85, 1.25)     # 时限扰动

HANDOFF_RADIUS_MIN_CM: float = 500.0  # handoff点采样最小半径（5米）
HANDOFF_RADIUS_MAX_CM: float = 1000.0  # handoff点采样最大半径（10米）


# ============================== FoodItem ==============================

@dataclass
class FoodItem:
    name: str
    category: str = ""
    odor: str = ""                       # 原 JSON: "odor"
    motion_sensitive: bool = False
    damage_level: int = 0                  # 0=完好，1=轻微损坏，2=中等损坏, 3=严重损坏
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

    odor_contamination: float = 0.0

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

    # —— 建筑边界信息 ——
    dropoff_building_box: Optional[Dict[str, Any]] = None

    # —— Handoff 相关字段 ——
    handoff_address: Optional[Vector] = None

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

    _lock: RLock = field(init=False, default_factory=RLock)

    # --- class-level id 生成锁 ---
    _id_counter: int = 0
    _id_lock: Lock = Lock()

    # --- preparation timeline (虚拟时间；接单时写入) ---
    prep_started_sim: Optional[float] = None
    prep_ready_sim: Optional[float] = None
    prep_longest_s: float = 0.0

    # --- 用于常规 prompt/提示 ---
    sim_started_s: float = 0.0
    sim_elapsed_active_s: float = 0.0
    sim_delivered_s: Optional[float] = None

    allowed_delivery_methods: Optional[List[str]] = None

    def __post_init__(self):
        # 线程安全分配 id
        with Order._id_lock:
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

        # 存储建筑边界信息
        self.dropoff_building_box = do_meta.get("building_box")

        if self.pickup_node is None or self.dropoff_node is None:
            raise Exception(False)

        self.road_name = (
            self.pickup_road_name
            if self.pickup_road_name == self.dropoff_road_name
            else f"{self.pickup_road_name} -> {self.dropoff_road_name}"
        )

    def _sample_handoff_position(self) -> Optional[Vector]:
        """在dropoff_node附近随机采样handoff位置，避免建筑内部"""
        if self.dropoff_node is None:
            return None
            
        center_x = float(self.dropoff_node.position.x)
        center_y = float(self.dropoff_node.position.y)
        
        # 最大尝试次数，避免无限循环
        max_attempts = 50
        
        for attempt in range(max_attempts):
            # 在圆形区域内随机采样，使用随机半径
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(HANDOFF_RADIUS_MIN_CM, HANDOFF_RADIUS_MAX_CM)
            
            handoff_x = center_x + radius * math.cos(angle)
            handoff_y = center_y + radius * math.sin(angle)
            handoff_pos = Vector(handoff_x, handoff_y)
            
            # 检查是否在建筑内部
            if not self._is_point_inside_building(handoff_pos):
                return handoff_pos
        
        # 如果所有尝试都失败，返回一个默认位置（建筑门口向外一定距离）
        # 计算从建筑中心到门口的方向向量
        if self.dropoff_building_box:
            building_center_x = self.dropoff_building_box.get("x", center_x)
            building_center_y = self.dropoff_building_box.get("y", center_y)
            
            # 从建筑中心到门口的方向
            door_direction_x = center_x - building_center_x
            door_direction_y = center_y - building_center_y
            
            # 归一化方向向量
            length = math.sqrt(door_direction_x**2 + door_direction_y**2)
            if length > 0:
                door_direction_x /= length
                door_direction_y /= length
                
                # 在门口外一定距离处放置handoff点，使用最小半径
                offset_distance = HANDOFF_RADIUS_MAX_CM
                handoff_x = center_x + door_direction_x * offset_distance
                handoff_y = center_y + door_direction_y * offset_distance
                return Vector(handoff_x, handoff_y)
        
        return None

    def _is_point_inside_building(self, point: Vector) -> bool:
        """检查点是否在建筑内部"""
        if self.dropoff_building_box is None:
            return False
            
        building_box = self.dropoff_building_box
        bx = building_box.get("x", 0.0)
        by = building_box.get("y", 0.0)
        bw = building_box.get("w", 0.0)
        bh = building_box.get("h", 0.0)
        byaw = building_box.get("yaw", 0.0)
        
        if bw <= 0 or bh <= 0:
            return False
        
        # 将点转换到建筑局部坐标系
        # 1. 平移到建筑中心
        local_x = point.x - bx
        local_y = point.y - by
        
        # 2. 旋转到建筑局部坐标系（反向旋转）
        yaw_rad = math.radians(-byaw)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        
        rotated_x = local_x * cos_yaw - local_y * sin_yaw
        rotated_y = local_x * sin_yaw + local_y * cos_yaw
        
        # 3. 检查是否在矩形内部
        half_w = bw / 2.0
        half_h = bh / 2.0
        
        return abs(rotated_x) <= half_w and abs(rotated_y) <= half_h

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
        # 读取大量状态，放在订单锁下以保证一致视图
        with self._lock:
            def _fmt_min(sec: float) -> str:
                s = max(0.0, float(sec))
                minutes = int((s + 59) // 60)  # Round up to minutes
                return f"{minutes} min"

            dist_m = float(self.distance_cm) / 100.0
            dist_km = dist_m / 1000.0

            px_m = float(self.pickup_node.position.x) / 100.0
            py_m = float(self.pickup_node.position.y) / 100.0
            dx_m = float(self.dropoff_node.position.x) / 100.0
            dy_m = float(self.dropoff_node.position.y) / 100.0

            tl = float(getattr(self, "time_limit_s", 0.0) or 0.0)
            earnings = float(getattr(self, "earnings", 0.0) or 0.0)

            lines = []

            if not bool(getattr(self, "is_accepted", False)):
                lines.extend([
                    f"[Order #{self.id}]",
                    f"  Pickup : ({px_m:.2f}m, {py_m:.2f}m) | road: {self.pickup_road_name}",
                    f"  Dropoff: ({dx_m:.2f}m, {dy_m:.2f}m) | road: {self.dropoff_road_name}",
                    f"  Dist   : {dist_m:.1f} m  ({dist_km:.3f} km)",
                    f"  Time Limit : {_fmt_min(tl)}",
                    f"  $$     : ${earnings:.2f}",
                ])
                if self.items:
                    items_str = ", ".join([f"{it.name} ({it.serving_temp_c:.0f}°C)" for it in self.items])
                    lines.append("  Items  : " + items_str)
                if self.special_note:
                    lines.append(f"  Note   : {self.special_note}")
                return "\n".join(lines)

            # accepted
            spent = float(getattr(self, "sim_elapsed_active_s", 0.0) or 0.0)
            left = tl - spent
            if tl > 0:
                if left >= 0:
                    time_line = f"  Time Left : {_fmt_min(left)}"
                else:
                    time_line = f"  OVERTIME  : {_fmt_min(-left)} — Deliver ASAP!"
            else:
                time_line = "  Time Left : N/A"

            if bool(getattr(self, "has_delivered", False)):
                status = "Delivered"
            else:
                picked = bool(getattr(self, "has_picked_up", False))
                ready = False
                if hasattr(self, "is_ready_for_pickup"):
                    try:
                        ready = bool(self.is_ready_for_pickup(self.sim_started_s + self.sim_elapsed_active_s))
                    except Exception:
                        ready = False

                if not picked:
                    if ready:
                        status = "Ready for pickup"
                    else:
                        remain_prep = 0.0
                        if hasattr(self, "remaining_prep_s"):
                            try:
                                remain_prep = float(self.remaining_prep_s(self.sim_started_s + self.sim_elapsed_active_s))
                            except Exception:
                                remain_prep = 0.0
                        status = f"Food is still being prepared (~{_fmt_min(remain_prep)})"
                else:
                    status = "Picked up, waiting for delivery"

            lines.extend([
                f"[Order #{self.id}]",
                f"  Pickup : ({px_m:.2f}m, {py_m:.2f}m) | road: {self.pickup_road_name}",
                f"  Dropoff: ({dx_m:.2f}m, {dy_m:.2f}m) | road: {self.dropoff_road_name}",
                time_line,
                f"  $$     : ${earnings:.2f}",
                f"  Status : {status}",
            ])
            if self.special_note:
                lines.append(f"  Note   : {self.special_note}")
            return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text()

    # ---------------- 给 Viewer 的简要 hint ----------------
    def to_active_order_hint(self) -> Dict[str, Any]:
        with self._lock:
            px, py = float(self.pickup_address.x), float(self.pickup_address.y)
            dx, dy = float(self.delivery_address.x), float(self.delivery_address.y)
            requires_handoff = 'hand_to_customer' in self.allowed_delivery_methods
            hint = dict(
                id=self.id,
                pickup_xy=(px, py),
                dropoff_xy=(dx, dy),
                distance_cm=self.distance_cm,
                eta_s=self.eta_s,
                earnings=self.earnings,
                road_name=self.road_name,
                requires_handoff=requires_handoff,
            )
            
            if requires_handoff and self.handoff_address:
                hx, hy = float(self.handoff_address.x), float(self.handoff_address.y)
                hint["handoff_xy"] = (hx, hy)
            
            return hint
    # ---------------- 备餐时间线 ----------------
    def active_now(self) -> float:
        return float(self.sim_started_s) + float(self.sim_elapsed_active_s or 0.0)

    def ready_at(self) -> float:
        # 统一基于活动时间轴：起点 + longest
        return float(self.sim_started_s) + float(self.prep_longest_s or 0.0)

    def is_ready_for_pickup(self, now_sim: Optional[float] = None) -> bool:
        with self._lock:
            if self.prep_started_sim is None:
                return False
            now = self.active_now() if now_sim is None else float(now_sim)
            return now >= self.ready_at()

    def remaining_prep_s(self, now_sim: Optional[float] = None) -> float:
        with self._lock:
            if self.prep_started_sim is None:
                return float(self.prep_longest_s or 0.0)
            now = self.active_now() if now_sim is None else float(now_sim)
            return max(0.0, self.ready_at() - now)



# =========================== OrderManager ===========================

class OrderManager:
    """
    固定容量订单池：
    - 只从 world_nodes 中随机抽样（pickup=restaurant；dropoff=building）
    - 只接受 JSON 字典列表菜单（menu = data["items"]）
    - 生成订单时随机分配 1~4 个菜品（每个菜品按 JSON 字段构建 FoodItem）
    """

    def __init__(self, capacity: int = 10, menu: Optional[List[Dict[str, Any]]] = None, clock: Optional[VirtualClock] = None, special_notes_map: Optional[Dict[str, List[str]]] = None, note_prob: float = 0.2):
        assert capacity > 0
        self.capacity = int(capacity)
        self._orders: List[Order] = []
        self._menu: List[Dict[str, Any]] = list(menu) if menu else []
        self._clock = clock if clock is not None else VirtualClock()
        self._lock: RLock = RLock()  # 管理器级锁（可重入，便于内部复用）
        self._special_notes_map: Dict[str, List[str]] = dict(special_notes_map or {})
        self._note_prob: float = float(note_prob)

    # ---------- 基础访问 ----------
    def __len__(self):
        with self._lock:
            return len(self._orders)

    def __iter__(self):
        # 迭代器返回快照，避免在遍历期间被修改
        with self._lock:
            return iter(list(self._orders))

    def list_orders(self) -> List[Order]:
        with self._lock:
            return list(self._orders)

    def get(self, order_id: int) -> Optional[Order]:
        with self._lock:
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
        if not isinstance(self._menu, list) or not self._menu:
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
            odor_contamination = 1.0 if odor == "strong" else 0.0

            items.append(FoodItem(
                name=name,
                category=category,
                odor=odor,
                odor_contamination=odor_contamination,
                motion_sensitive=motion_sensitive,
                nonthermal_time_sensitive=nonthermal_time_sensitive,
                prep_time_s=prep_time_s,
                serving_temp_c=serving_temp_c,
                safe_min_c=safe_min_c,
                safe_max_c=safe_max_c,
                heat_capacity=heat_capacity,
                temp_c=float("nan"),
            ))

        return items

    # ---------- 生成单个订单 ----------
    def _spawn_one_order(self, city_map: Any, world_nodes: List[Dict[str, Any]], _ue = None) -> Order:
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
        # 初始化路线与计价放在订单内部（单订单内无需管 OM 锁）
        order.compute()

        longest = 0
        if items:
            longest = max(int(getattr(it, "prep_time_s", 0) or 0) for it in items)
        order.prep_longest_s = float(longest)

        if self._special_notes_map and random.random() < self._note_prob:
            phrase, methods = random.choice(list(self._special_notes_map.items()))
            order.special_note = phrase
            order.allowed_delivery_methods = list(methods or [])
        else:
            order.special_note = ""                # 显示为空串
            order.allowed_delivery_methods = []  # None 表示四种都行

        if 'hand_to_customer' in order.allowed_delivery_methods:
            handoff_address = order._sample_handoff_position()
            if handoff_address is not None:
                order.handoff_address = handoff_address
            else:
                order.allowed_delivery_methods.remove('hand_to_customer')

        if 'hand_to_customer' in order.allowed_delivery_methods and _ue:
             _ue.spawn_customer(order.id, order.handoff_address.x, order.handoff_address.y)

        return order

    # ---------- 填充到固定容量 ----------
    def fill_pool(self, city_map: Any, world_nodes: List[Dict[str, Any]], _ue = None):
        with self._lock:
            while len(self._orders) < self.capacity:
                self._orders.append(self._spawn_one_order(city_map, world_nodes, _ue))

    # ---------- 维护池：完成/移除并补齐 ----------
    def complete_order(self, order_id: int, city_map: Any, world_nodes: List[Dict[str, Any]], _ue = None):
        with self._lock:
            target = None
            for o in self._orders:
                if o.id == order_id:
                    target = o
                    break
            if target is None:
                return
            # 标记完成放在订单锁下
            with target._lock:
                target.has_delivered = True
            # 从池中移除并补齐
            self._orders = [o for o in self._orders if o.id != order_id]
            # fill_pool 内部会再次获取 OM 锁（RLock 可重入）
            self.fill_pool(city_map, world_nodes, _ue)

    def remove_order(self, order_ids: Union[int, Iterable[int]], city_map: Any, world_nodes: List[Dict[str, Any]], _ue = None):
        if isinstance(order_ids, int):
            ids = {int(order_ids)}
        else:
            ids = {int(i) for i in order_ids}
        if not ids:
            return
        with self._lock:
            self._orders = [o for o in self._orders if int(o.id) not in ids]
            self.fill_pool(city_map, world_nodes, _ue)

    def accept_order(self, order_ids: Union[int, Iterable[int]]):
        """
        - 传 int：返回 bool
        - 传 Iterable[int]：返回 (accepted_ids: List[int], failed_ids: List[int])
        仅设置接单与时间线；不从池子移除（移除交给 remove_order）
        """
        def _accept_one(oid: int) -> bool:
            with self._lock:
                o = self.get(int(oid))
                if o is None:
                    return False
                # 订单内部状态更新放在订单锁下
                with o._lock:
                    if getattr(o, "is_accepted", False):
                        return False
                    o.is_accepted = True
                    # 接单时间线（用 OM 的 clock，如果没有就用真实时间兜底）
                    now = float(self._clock.now_sim() if hasattr(self, "_clock") and self._clock else time.time())
                    o.sim_started_s = now
                    o.sim_elapsed_active_s = 0.0
                    o.sim_delivered_s = None
                    longest = float(getattr(o, "prep_longest_s", 0.0) or 0.0)
                    o.prep_started_sim = now
                    o.prep_ready_sim = now + longest
                    return True

        # 单个 id：直接返回布尔
        if isinstance(order_ids, int):
            return _accept_one(int(order_ids))

        # 多个 id：去重后分别接单，返回 (accepted_ids, failed_ids)
        accepted_ids: List[int] = []
        failed_ids: List[int] = []
        seen = set()

        for oid in order_ids:
            i = int(oid)
            if i in seen:
                continue
            seen.add(i)
            if _accept_one(i):
                accepted_ids.append(i)
            else:
                failed_ids.append(i)

        return accepted_ids, failed_ids

    def recompute_all(self):
        # 快照后逐个在订单锁下重算，避免长时间占用 OM 锁
        with self._lock:
            snapshot = list(self._orders)
        for o in snapshot:
            with o._lock:
                o.compute()

    # ---------- 一键文本输出 ----------
    def orders_text(self) -> str:
        with self._lock:
            snapshot = list(self._orders)
        if not snapshot:
            return "No orders."
        lines: List[str] = []
        lines.append(f"Orders ({len(snapshot)}):")
        for o in snapshot:
            # 保证 to_text 在订单锁下生成一致视图
            lines.append(o.to_text())
            lines.append("-" * 60)
        return "\n".join(lines)

    # ---------- 输出给 MapViewer（兼容） ----------
    def to_active_order_hints(self) -> List[Dict[str, Any]]:
        with self._lock:
            snapshot = list(self._orders)
        # 在订单锁下读取各自 hint
        hints: List[Dict[str, Any]] = []
        for o in snapshot:
            hints.append(o.to_active_order_hint())
            # 追加道路信息
            hints[-1]["pickup_road"] = o.pickup_road_name
            hints[-1]["dropoff_road"] = o.dropoff_road_name
        return hints

    def status_text(self) -> str:
        return "Active"