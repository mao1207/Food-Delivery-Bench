# -*- coding: utf-8 -*-
# Base/Comms.py
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Dict, Any, List, Optional, Tuple

try:
    from Base.Timer import VirtualClock
except Exception:
    class VirtualClock:
        def now_sim(self) -> float:
            return time.time()

# 需要直接实例化保温袋（对方没有时创建一个）
from Base.Insulated_bag import InsulatedBag


# ====== 请求类型（先留 4 个） ======
class HelpType(str, Enum):
    HELP_PICKUP   = "HELP_PICKUP"
    HELP_DELIVERY = "HELP_DELIVERY"
    HELP_BUY      = "HELP_BUY"
    HELP_CHARGE   = "HELP_CHARGE"


@dataclass
class HelpRequest:
    req_id: int
    publisher_id: str
    kind: HelpType
    reward: float
    time_limit_s: float
    created_sim: float
    # 规范化参数（避免 agent 自己写坐标）
    order_id: Optional[int] = None
    buy_items: Optional[Dict[str, int]] = None
    target_pct: Optional[float] = None
    # 展示
    brief: str = ""
    location_xy: Optional[Tuple[float, float]] = None
    # 状态
    accepted_by: Optional[str] = None   # 接单者


class CommsSystem:
    """
    极简通信系统：
    - 求助板：发布/接受/修改
    - 物品转交（核心放在这里，DeliveryMan 只需调用）：
        * inventory：A 扣、B 加
        * escooter：对象引用移交（不改 owner_id；能否骑由 EScooter/DM 校验）
        * bag_items：按 bag_cmd 直接装入对方保温袋（支持多订单）
          - 要求给出“订单 -> (items, bag_cmd)”的映射，bag_cmd 可写成：
                '1,2 -> A; 3 -> B' 或 'order 12: 1,2 -> A; 3 -> B'
          - 本实现不做回滚；中途失败会抛错（符合“报错就报、回头改”的诉求）
    """
    def __init__(self, clock: Optional[VirtualClock] = None):
        self.clock = clock if clock is not None else VirtualClock()
        self._lock = Lock()
        self._next_req_id = 1
        self._board: Dict[int, HelpRequest] = {}      # 未被接单的请求
        self._active: Dict[int, HelpRequest] = {}     # 已被接单但未结算（后续迭代用）
        self._agents: Dict[str, Any] = {}             # agent_id -> DeliveryMan

    # ---------- Agent 注册 ----------
    def register_agent(self, dm: Any):
        with self._lock:
            self._agents[str(dm.agent_id)] = dm

    def get_agent(self, agent_id: str) -> Optional[Any]:
        return self._agents.get(str(agent_id))

    # ---------- 求助板：发布 ----------
    def post_request(
        self,
        publisher_id: str,
        kind: HelpType,
        *,
        reward: float,
        time_limit_s: float,
        order_id: Optional[int] = None,
        buy_items: Optional[Dict[str, int]] = None,
        target_pct: Optional[float] = None,
        brief: Optional[str] = None,
        location_xy: Optional[Tuple[float, float]] = None,
    ) -> Tuple[bool, str, Optional[int]]:
        now = self.clock.now_sim()
        with self._lock:
            rid = self._next_req_id
            self._next_req_id += 1
            req = HelpRequest(
                req_id=rid,
                publisher_id=str(publisher_id),
                kind=HelpType(kind),
                reward=float(reward),
                time_limit_s=float(time_limit_s),
                created_sim=float(now),
                order_id=int(order_id) if order_id is not None else None,
                buy_items=dict(buy_items) if buy_items else None,
                target_pct=float(target_pct) if target_pct is not None else None,
                brief=str(brief) if brief else self._auto_brief(kind, order_id, buy_items, target_pct),
                location_xy=location_xy
            )
            self._board[rid] = req
            return True, "posted", rid

    # ---------- 求助板：接受（接单后从板上删除并放入 active） ----------
    def accept_request(self, helper_id: str, req_id: int) -> Tuple[bool, str]:
        with self._lock:
            req = self._board.get(int(req_id))
            if not req:
                return False, "not found or already accepted"
            req.accepted_by = str(helper_id)
            self._active[req.req_id] = req
            self._board.pop(req.req_id, None)
            return True, "accepted"

    # ---------- 求助板：修改（仅发布者 & 未被接单） ----------
    def modify_request(self, publisher_id: str, req_id: int,
                       *, reward: Optional[float] = None,
                       time_limit_s: Optional[float] = None) -> Tuple[bool, str]:
        with self._lock:
            req = self._board.get(int(req_id))
            if not req:
                return False, "not found or already accepted"
            if req.publisher_id != str(publisher_id):
                return False, "permission denied"
            if reward is not None:
                req.reward = float(reward)
            if time_limit_s is not None:
                req.time_limit_s = float(time_limit_s)
            return True, "modified"

    # ---------- 求助板：查看 ----------
    def list_board(self) -> List[Dict[str, Any]]:
        now = self.clock.now_sim()
        with self._lock:
            out = []
            for r in self._board.values():
                remain = max(0.0, (r.created_sim + r.time_limit_s) - now)
                out.append(dict(
                    id=r.req_id, publisher=r.publisher_id, kind=r.kind.value,
                    brief=r.brief, reward=r.reward, time_left_s=remain,
                    location_xy=r.location_xy
                ))
            return out

    # ---------- 对外：统一“给东西”入口 ----------
    def give(self, from_id: str, to_id: str, payload: Dict[str, Any]) -> Tuple[bool, str]:
        """
        payload 形式：
        - {"type":"inventory", "items":{"energy_drink":2, "escooter_battery_pack":1}}
        - {"type":"escooter"}
        - {"type":"bag_items", "plan": {  # 推荐：支持多订单
              12: {"items":[...FoodItem...], "bag_cmd":"1,2 -> A; 3 -> B"},
              18: {"items":[...],            "bag_cmd":"1 -> B"}
            }}
          兼容旧形式（单订单）：
        - {"type":"bag_items", "order_id":12, "items":[...], "bag_cmd":"order 12: 1,2 -> A; 3 -> B"}
        """
        a = self.get_agent(from_id)
        b = self.get_agent(to_id)
        if a is None or b is None:
            return False, "agent not found"

        typ = str(payload.get("type", "")).lower()
        if typ == "inventory":
            items = dict(payload.get("items") or {})
            return self._give_inventory(a, b, items)

        if typ == "escooter":
            return self._give_escooter(a, b)

        if typ == "bag_items":
            if "plan" in payload:
                return self._give_bag_items_plan(a, b, payload["plan"])
            # 兼容单订单
            order_id = int(payload["order_id"])
            items = list(payload["items"])
            bag_cmd = str(payload.get("bag_cmd", "")).strip()
            plan = {order_id: {"items": items, "bag_cmd": bag_cmd}}
            return self._give_bag_items_plan(a, b, plan)

        return False, "unknown give type"

    # ---------- 对外：便捷封装（可给 DeliveryMan 直接用） ----------
    def give_inventory(self, from_id: str, to_id: str, items: Dict[str, int]) -> Tuple[bool, str]:
        a = self.get_agent(from_id); b = self.get_agent(to_id)
        if a is None or b is None:
            return False, "agent not found"
        return self._give_inventory(a, b, items)

    def give_escooter(self, from_id: str, to_id: str) -> Tuple[bool, str]:
        a = self.get_agent(from_id); b = self.get_agent(to_id)
        if a is None or b is None:
            return False, "agent not found"
        return self._give_escooter(a, b)

    def give_bag_items_cmd(self, from_id: str, to_id: str,
                           plan: Dict[int, Dict[str, Any]]) -> Tuple[bool, str]:
        """
        plan: { order_id: {"items":[FoodItem,...], "bag_cmd":"1,2 -> A; 3 -> B" 或 "order 12: ..."} }
        """
        a = self.get_agent(from_id); b = self.get_agent(to_id)
        if a is None or b is None:
            return False, "agent not found"
        return self._give_bag_items_plan(a, b, plan)

    # ====== 内部：inventory 互转 ======
    def _give_inventory(self, a: Any, b: Any, items: Dict[str, int]) -> Tuple[bool, str]:
        inv_a = getattr(a, "inventory", None)
        inv_b = getattr(b, "inventory", None)
        if inv_a is None or inv_b is None:
            return False, "inventory unavailable"

        # 校验
        for k, v in (items or {}).items():
            q = int(v)
            if q <= 0:
                return False, f"invalid qty for {k}"
            have = int(inv_a.get(k, 0))
            if have < q:
                return False, f"not enough {k} (have {have})"

        # 原子区：扣/加（不回滚）
        with self._lock:
            for k, v in items.items():
                q = int(v)
                inv_a[k] = int(inv_a.get(k, 0)) - q
                inv_b[k] = int(inv_b.get(k, 0)) + q
        return True, "ok"

    # ====== 内部：escooter 转交（不改 owner_id） ======
    def _give_escooter(self, a: Any, b: Any) -> Tuple[bool, str]:
        es_a = getattr(a, "e_scooter", None)
        es_b = getattr(b, "e_scooter", None)
        if es_a is None:
            return False, "source has no scooter"
        if es_b is not None:
            return False, "target already has a scooter"

        with self._lock:
            b.e_scooter = es_a
            a.e_scooter = None
        return True, "ok"

    # ====== 内部：保温袋内食物转交（按 bag_cmd 直接放入对方袋） ======
    def _give_bag_items_plan(self, a: Any, b: Any,
                             plan: Dict[int, Dict[str, Any]]) -> Tuple[bool, str]:
        """
        plan 示例：
            {
              12: {"items":[...FoodItem...], "bag_cmd":"1,2 -> A; 3 -> B"},
              18: {"items":[...],            "bag_cmd":"order 18: 1 -> B"}
            }
        约定：
          - items 的顺序就是“1..n”的编号顺序
          - bag_cmd 可是不含 'order x:' 的“尾部命令”，也可包含 'order x:' 前缀
        """
        bag_a = getattr(a, "insulated_bag", None)
        if bag_a is None or not hasattr(bag_a, "remove_items"):
            return False, "source has no bag or remove_items"

        bag_b = getattr(b, "insulated_bag", None)
        if bag_b is None:
            b.insulated_bag = InsulatedBag()
            bag_b = b.insulated_bag

        if not hasattr(bag_b, "add_items"):
            return False, "target bag has no add_items"

        with self._lock:
            # 1) 先从 A 的袋里删（不回滚）
            all_items: List[Any] = []
            for _, obj in plan.items():
                all_items += list(obj.get("items") or [])
            if all_items:
                bag_a.remove_items(all_items)

            # 2) 再按订单逐个装入 B 的袋里
            for oid, obj in plan.items():
                items = list(obj.get("items") or [])
                cmd_raw = str(obj.get("bag_cmd", "")).strip()
                if not items:
                    continue
                # 规范化：取“order <id>:”后的尾部命令（也允许直接给尾部命令）
                per_order_cmd = self._normalize_per_order_cmd(cmd_raw, int(oid))
                # 1..n -> item
                items_map = {i + 1: items[i] for i in range(len(items))}
                # 按命令把这些“新物品”放入对应隔层
                bag_b.add_items(per_order_cmd, items_map)

        return True, "ok"

    # ---------- 小工具 ----------
    def _normalize_per_order_cmd(self, raw: str, oid: int) -> str:
        """
        接受两种形式：
          - '1,2 -> A; 3 -> B'
          - 'order 12: 1,2 -> A; 3 -> B'
        返回“尾部命令”（不含 'order x:' 前缀）
        """
        s = (raw or "").strip()
        if not s:
            return ""
        low = s.lower()
        if low.startswith("order "):
            p = s.find(":")
            if p != -1:
                # 可选地校验一下 id（即使不一致也不强制报错，保持宽容）
                # head = s[:p].strip()  # 'order 12'
                return s[p + 1:].strip()
        return s

    def _auto_brief(self, kind: HelpType, order_id, buy_items, target_pct) -> str:
        if kind == HelpType.HELP_PICKUP:
            return f"Help pick up order #{order_id}"
        if kind == HelpType.HELP_DELIVERY:
            return f"Help deliver order #{order_id}"
        if kind == HelpType.HELP_BUY:
            parts = []
            if buy_items:
                for k, v in buy_items.items():
                    parts.append(f"{k} x{int(v)}")
            return "Help buy: " + (", ".join(parts) if parts else "(none)")
        if kind == HelpType.HELP_CHARGE:
            t = target_pct if target_pct is not None else 100
            return f"Help charge e-scooter to {float(t):.0f}%"
        return kind.value


# ====== 单例（便于在 main / DeliveryMan 里轻松拿到） ======
_COMMS_SINGLETON: Optional[CommsSystem] = None

def init_comms(clock: Optional[VirtualClock] = None) -> CommsSystem:
    global _COMMS_SINGLETON
    _COMMS_SINGLETON = CommsSystem(clock=clock)
    return _COMMS_SINGLETON

def get_comms() -> Optional[CommsSystem]:
    return _COMMS_SINGLETON