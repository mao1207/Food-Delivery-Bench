# -*- coding: utf-8 -*-
"""
ActionSpace.py
- ACTION_API_SPEC：动作空间/输出格式说明（供模型参考，不替代 system prompt）
- sanitize_model_text(text): 提取模型的一行函数调用
- parse_action(text, dm): 解析为 DMAction（严格与 Base/DeliveryMan.py 对齐）
- action_to_text(action, dm=None): 将 DMAction 转为简洁英文描述（含关键信息）

约定（与 DeliveryMan 保持一致）：
- MOVE(x, y): 坐标以“米”为单位，参数必须带 m 后缀（如 102.3m），解析后统一为厘米。
- RENT_CAR(avg_speed_m_s=...): 入参 m/s，解析后换算为 cm/s。
- EDIT_HELP(new_ttl_min=...): 入参分钟，解析后换算为秒。
- PICKUP(orders=[...]): 仅“取餐到手”；放入保温袋需另行调用 PLACE_FOOD_IN_BAG(bag_cmd=...).

新增：
- DROP_OFF(oid=<int>, method="leave_at_door|knock|call|hand_to_customer")
"""

from __future__ import annotations
import ast
import json
import re
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "ACTION_API_SPEC",
    "sanitize_model_text",
    "parse_action",
    "action_to_text",
]

# =========================================================
# 1) 规范文本
# =========================================================
ACTION_API_SPEC: str = r"""
You must output EXACTLY ONE action per turn, as a ONE-LINE function call (no prose, no code fences, no comments).

COMMANDS (UPPERCASE):
- VIEW_ORDERS()
- VIEW_BAG()
- ACCEPT_ORDER(order_id) or ACCEPT_ORDER([order_id, ...])
- MOVE(x, y) # x,y in meters; MUST use 'm' suffix (e.g., MOVE(102.3m, -5.0m))
- MOVE(x, y, pace="accel"|"normal"|"decel")
- PICKUP(orders=[12, 18])  # pick ready orders at the pickup door (two-step: pickup -> place into bag)
- PLACE_FOOD_IN_BAG(bag_cmd="order 12: 1,2 -> A; 3 -> B")
- CHARGE(target_pct=100)
- WAIT(seconds=NN) or WAIT("charge_done")
- REST(target_pct=100)
- BUY(item_id="energy_drink", qty=1) or BUY(items=[{"item_id":"energy_drink","qty":2}, {"name":"escooter_battery_pack","qty":1}])
- USE_BATTERY_PACK()
- USE_ENERGY_DRINK()
- USE_ICE_PACK(comp="A")   # instantly set compartment A to 0°C; consumes one ice_pack
- USE_HEAT_PACK(comp="B")  # instantly set compartment B to 60°C; consumes one heat_pack
- SWITCH(to="walk"|"e-scooter"|"car" | "drag_scooter")
- RENT_CAR(rate_per_min=1.0, avg_speed_m_s=20)  # speed in m/s
- RETURN_CAR()
- VIEW_HELP_BOARD()
- POST_HELP(kind="HELP_PICKUP"|"HELP_DELIVERY"|"HELP_BUY"|"HELP_CHARGE", bounty=5.0, ttl_s=300, payload={...})
- ACCEPT_HELP(req_id=123)
- EDIT_HELP(req_id=123, new_bounty=6.0, new_ttl_min=10)  # duration in minutes
- PLACE_TEMP_BOX(req_id=123, location=(110.0m, 445.0m), content={"inventory": {"energy_drink": 1}, "food": "", "escooter": ""})
- TAKE_FROM_TEMP_BOX(req_id=123)
- REPORT_HELP_FINISHED(req_id=123)
- DROP_OFF(oid=<int>, method="leave_at_door|knock|call|hand_to_customer")
- SAY("text")  # broadcast to all
- SAY(to="agent_id", text="text")  # direct message; to can be "ALL" or "*"

PRECONDITIONS (obey strictly):
- PICKUP only at the pickup door of the pick up address.
- PLACE_FOOD_IN_BAG only after you have pending food items in hand.
- CHARGE only at a charging_station; the scooter may be parked automatically.
- REST only at a rest_area.
- SWITCH to "e-scooter"/"car" only when near your vehicle.
- For deliveries, MOVE to the dropoff first, then call DROP_OFF(oid, method=...).
- VIEW_ORDERS can be used to check available orders. If the current context already contains detailed order information, you do not need to repeat VIEW_ORDERS unnecessarily.
- For HELP_BUY / HELP_PICKUP / HELP_CHARGE: after completing the task, PLACE_TEMP_BOX at deliver_xy and REPORT_HELP_FINISHED. The original agent may later TAKE_FROM_TEMP_BOX at that location. For HELP_DELIVERY or HELP_CHARGE requests *you post*, you must PLACE_TEMP_BOX with your items/vehicle at provide_xy and let others TAKE_FROM_TEMP_BOX from there.
  (bounty = money reward in dollars, ttl_s = expiration in seconds, provide_xy = give from here, deliver_xy = deliver to here, target_pct = target battery %)
- When you are carrying items that are delicate, spill-prone, easily deformed, or otherwise require gentle handling, note that moving with pace="accel" may cause some damage to the food.


OUTPUT EXAMPLES (exactly one line):
  VIEW_ORDERS()
  VIEW_BAG()
  ACCEPT_ORDER(12)
  ACCEPT_ORDER([12, 18])
  MOVE(102.3m, 885.5m)
  MOVE(250.0m, 120.0m, pace="accel")
  PICKUP(orders=[12])
  PLACE_FOOD_IN_BAG(bag_cmd="order 12: 1,2 -> A; 3 -> B")
  CHARGE(target_pct=100)
  WAIT("charge_done")
  BUY(item="energy_drink", qty=1)
  BUY(items=[{"item_id":"energy_drink","qty":2}, {"name":"escooter_battery_pack","qty":1}])
  SWITCH(to="e-scooter")
  RENT_CAR(rate_per_min=1.0, avg_speed_m_s=22.0)
  VIEW_HELP_BOARD()
  PLACE_TEMP_BOX(req_id=77, location=(110.0m, 445.0m), content={"food": "", "inventory": {"energy_drink": 1}})
  POST_HELP(kind="HELP_PICKUP", bounty=5.0, ttl_s=600, payload={"order_id": 21, "deliver_xy": (205.0m, 318.0m)})
  POST_HELP(kind="HELP_DELIVERY", bounty=6.0, ttl_s=600, payload={"order_id": 18, "provide_xy": (110.0m, 445.0m)})
  POST_HELP(kind="HELP_BUY", bounty=8.0, ttl_s=900, payload={"buy_list": [("energy_drink", 2), ("escooter_battery_pack", 1)], "deliver_xy": (300.0m, 500.0m)})
  POST_HELP(kind="HELP_CHARGE", bounty=7.5, ttl_s=900, payload={"provide_xy": (120.0m, 140.0m), "deliver_xy": (400.0m, 600.0m), "target_pct": 80})
  DROP_OFF(oid=12, method="leave_at_door")
  USE_ICE_PACK(comp="A")
  USE_HEAT_PACK(comp="B")
  SAY("Hi all, anyone near the charging station?")
  SAY(to="7", text="I'll take order #12, meet you at the door.")
"""

# =========================================================
# 2) 解析辅助
# =========================================================
_CALL_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", re.DOTALL)
FENCE_RE = re.compile(r"^\s*```.*?\n(.*?)\n```", re.DOTALL)

_CANON = {
    "MOVE": "MOVE",
    "MOVE_TO": "MOVE",

    "VIEW_ORDERS": "VIEW_ORDERS",

    "ACCEPT_ORDER": "ACCEPT_ORDER",

    "PICKUP": "PICKUP",

    "PLACE_FOOD_IN_BAG": "PLACE_FOOD_IN_BAG",
    "PLACE_IN_BAG": "PLACE_FOOD_IN_BAG",
    "PLACE_BAG": "PLACE_FOOD_IN_BAG",
    "BAG": "PLACE_FOOD_IN_BAG",

    "CHARGE": "CHARGE",
    "CHARGE_ESCOOTER": "CHARGE",

    "WAIT": "WAIT",

    "REST": "REST",

    "BUY": "BUY",

    "USE_BATTERY_PACK": "USE_BATTERY_PACK",
    "USE_ENERGY_DRINK": "USE_ENERGY_DRINK",

    "SWITCH": "SWITCH",
    "SWITCH_TRANSPORT": "SWITCH",

    "RENT_CAR": "RENT_CAR",
    "RETURN_CAR": "RETURN_CAR",

    "VIEW_HELP_BOARD": "VIEW_HELP_BOARD",

    "POST_HELP": "POST_HELP",
    "POST_HELP_REQUEST": "POST_HELP",

    "ACCEPT_HELP": "ACCEPT_HELP",
    "ACCEPT_HELP_REQUEST": "ACCEPT_HELP",

    "EDIT_HELP": "EDIT_HELP",
    "EDIT_HELP_REQUEST": "EDIT_HELP",

    "PLACE_TEMP_BOX": "PLACE_TEMP_BOX",
    "DROP_TEMP_BOX": "PLACE_TEMP_BOX",

    "TAKE_FROM_TEMP_BOX": "TAKE_FROM_TEMP_BOX",
    "TAKE_TEMP_BOX": "TAKE_FROM_TEMP_BOX",

    "REPORT_HELP_FINISHED": "REPORT_HELP_FINISHED",
    "REPORT_DONE": "REPORT_HELP_FINISHED",

    "VIEW_BAG": "VIEW_BAG",
    "USE_ICE_PACK": "USE_ICE_PACK",
    "USE_HEAT_PACK": "USE_HEAT_PACK",

    # --- NEW: manual drop-off
    "DROP_OFF": "DROP_OFF",
    "DELIVER": "DROP_OFF",
    "DELIVER_ORDER": "DROP_OFF",

    "SAY": "SAY",
}

def sanitize_model_text(text: str) -> str:
    """去掉围栏/多余行，仅保留形如 NAME(... ) 的首行。"""
    t = (text or "").strip()
    m = FENCE_RE.search(t)
    if m:
        t = m.group(1).strip()
    t = t.splitlines()[0].strip()
    t = t.rstrip(";，。.")
    return t

# —— 将不在字符串内的 “<number>m” 替换为厘米数字
_UNIT_NUM_M = re.compile(r'(?P<num>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*m\b')

def _convert_m_to_cm_in_args(s: str) -> str:
    out: List[str] = []
    i, n = 0, len(s)
    in_str, quote, esc = False, None, False
    while i < n:
        ch = s[i]
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            else:
                if ch == '\\':
                    esc = True
                elif ch == quote:
                    in_str, quote = False, None
            i += 1
            continue
        m = _UNIT_NUM_M.match(s, i)
        if m:
            num_m = float(m.group('num'))
            num_cm = num_m * 100.0
            out.append(str(int(round(num_cm))) if abs(num_cm - round(num_cm)) < 1e-9 else f"{num_cm:.6f}".rstrip('0').rstrip('.'))
            i = m.end()
            continue
        if ch in ('"', "'"):
            in_str, quote = True, ch
        out.append(ch)
        i += 1
    return ''.join(out)

def _ast_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)) and isinstance(node.operand, ast.Constant):
        v = node.operand.value
        if isinstance(v, (int, float)):
            return +v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.Dict):
        return {_ast_value(k): _ast_value(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.List):
        return [_ast_value(e) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_ast_value(e) for e in node.elts)
    if isinstance(node, ast.Name) and node.id in ("True", "False", "None"):
        return {"True": True, "False": False, "None": None}[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _ast_value(node.left); right = _ast_value(node.right)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
    raise ValueError(f"Unsupported expression: {ast.dump(node, include_attributes=False)}")

def _parse_call(text: str) -> Tuple[str, List[Any], Dict[str, Any]]:
    m = _CALL_RE.match(text)
    if not m:
        raise ValueError("Output is not a function call like NAME(...).")
    raw_name = m.group(1).strip()
    name = _CANON.get(raw_name.upper(), raw_name.upper())
    args_src = m.group(2).strip()
    if args_src == "":
        return name, [], {}
    args_src = _convert_m_to_cm_in_args(args_src)  # 单位换算
    fake = f"__F__({args_src})"
    tree = ast.parse(fake, mode="eval")
    if not isinstance(tree.body, ast.Call):
        raise ValueError("Not a call.")
    call = tree.body
    pos = [_ast_value(a) for a in call.args]
    kw = {kw.arg: _ast_value(kw.value) for kw in call.keywords if kw.arg is not None}
    return name, pos, kw

# —— 工具：取当前取餐口的活跃订单
def _orders_at_pickup(dm, tol_cm: float = 300.0) -> List[Any]:
    res = []
    for o in getattr(dm, "active_orders", []) or []:
        if getattr(o, "has_picked_up", False):
            continue
        node = getattr(o, "pickup_node", None)
        if node is None:
            continue
        try:
            px, py = float(node.position.x), float(node.position.y)
        except Exception:
            continue
        if dm._is_at_xy(px, py, tol_cm=tol_cm):
            res.append(o)
    return res

def _lookup_order(dm, oid: int) -> Any:
    om = getattr(dm, "_order_manager", None)
    oid = int(oid)

    # 主路径：强约束 OM 实现 get(int)
    get = getattr(om, "get", None) if om is not None else None
    if callable(get):
        order = get(oid)
        if order is not None:
            return order
    if om is not None:
        if hasattr(om, "get"):
            o = getattr(om, "get")(oid)  # type: ignore
            if o is not None:
                return o
        pool = list(getattr(om, "_orders", []) or [])
        for o in pool:
            if getattr(o, "id", None) == oid:
                return o
    for o in getattr(dm, "active_orders", []) or []:
        if getattr(o, "id", None) == oid:
            return o
    # also look up attached helper orders
    o = (getattr(dm, "help_orders", {}) or {}).get(int(oid))
    if o is not None:
        return o
    
    return None

# =========================================================
# 3) 解析为 DMAction
# =========================================================
def parse_action(model_text: str, dm: Any):
    """
    把 VLM 的一行输出解析为 DMAction。
    成功返回 DMAction；失败抛 ValueError（由上层记录错误）。
    """
    from Base.DeliveryMan import DMAction, DMActionKind  # 延迟导入避免循环

    text = sanitize_model_text(model_text)
    name, pos, kw = _parse_call(text)

    if name == "MOVE":
        if len(pos) >= 2:
            x, y = float(pos[0]), float(pos[1])
        else:
            x, y = float(kw.get("x")), float(kw.get("y"))

        data = dict(tx=x, ty=y, use_route=True, snap_cm=120.0)

        # 可选 pace
        p = kw.get("pace", None)
        if isinstance(p, str):
            p_norm = p.strip().lower()
            if p_norm in ("accel", "normal", "decel"):
                data["pace"] = p_norm

        return DMAction(DMActionKind.MOVE_TO, data=data)
        

    if name == "VIEW_ORDERS":
        return DMAction(DMActionKind.VIEW_ORDERS, data={})

    if name == "ACCEPT_ORDER":
        # 兼容：ACCEPT_ORDER(12) / ACCEPT_ORDER([12,18]) / ACCEPT_ORDER(ids=[...]) / ACCEPT_ORDER(order_id=12)
        oids: List[int] = []
        oid: Optional[int] = None

        if len(pos) >= 1:
            arg0 = pos[0]
            if isinstance(arg0, (list, tuple)):
                oids = [int(v) for v in arg0]
            else:
                oid = int(arg0)
        elif "ids" in kw and isinstance(kw["ids"], (list, tuple)):
            oids = [int(v) for v in kw["ids"]]
        else:
            oid = int(kw.get("order_id") or kw.get("id"))

        if oids:
            return DMAction(DMActionKind.ACCEPT_ORDER, data=dict(oids=oids))
        return DMAction(DMActionKind.ACCEPT_ORDER, data=dict(oid=int(oid)))

    if name == "PICKUP":
        # 优先使用显式 orders=[...]; 否则默认取“当前取餐口可取的订单”
        oids: List[int] = []
        if len(pos) >= 1 and isinstance(pos[0], (list, tuple)):
            oids = [int(v) for v in pos[0]]
        elif "orders" in kw and isinstance(kw["orders"], (list, tuple)):
            oids = [int(v) for v in kw["orders"]]
        orders: List[Any] = []
        if oids:
            for oid in oids:
                o = _lookup_order(dm, int(oid))
                if o is not None:
                    orders.append(o)
        else:
            orders = _orders_at_pickup(dm, tol_cm=300.0)
        print(f"DEBUG: PICKUP orders = {[getattr(o, 'id', None) for o in orders]}")
        if not orders:
            raise ValueError("PICKUP: no ready orders (MOVE to the pickup door or specify orders=[...]).")
        return DMAction(DMActionKind.PICKUP, data=dict(orders=orders, tol_cm=300.0))

    if name == "PLACE_FOOD_IN_BAG":
        bag_cmd = (pos[0] if (len(pos) >= 1 and isinstance(pos[0], str)) else kw.get("bag_cmd") or kw.get("cmd") or "").strip()
        if not bag_cmd:
            raise ValueError("PLACE_FOOD_IN_BAG needs bag_cmd.")
        return DMAction(DMActionKind.PLACE_FOOD_IN_BAG, data=dict(bag_cmd=bag_cmd))

    if name == "CHARGE":
        target = float(pos[0]) if len(pos) >= 1 else float(kw.get("target_pct", 100.0))
        return DMAction(DMActionKind.CHARGE_ESCOOTER, data=dict(target_pct=target))

    if name == "WAIT":
        if len(pos) >= 1 and isinstance(pos[0], str) and pos[0].strip().lower() in ("charge_done", "done", "charge"):
            return DMAction(DMActionKind.WAIT, data=dict(until="charge_done"))
        if "until" in kw and str(kw["until"]).lower() == "charge_done":
            return DMAction(DMActionKind.WAIT, data=dict(until="charge_done"))
        secs = float(pos[0]) if len(pos) >= 1 else float(kw.get("seconds", kw.get("duration_s", 0.0)))
        return DMAction(DMActionKind.WAIT, data=dict(duration_s=secs))

    if name == "REST":
        tgt = float(pos[0]) if len(pos) >= 1 else float(kw.get("target_pct", 100.0))
        return DMAction(DMActionKind.REST, data=dict(target_pct=tgt))

    if name == "BUY":
        # 两种形态：
        # 1) BUY(item_id="energy_drink", qty=2) / BUY(name="...", qty=2) / BUY(item="...", qty=2)
        # 2) BUY(items=[{"item_id":"energy_drink","qty":2}, {"name":"escooter_battery_pack","qty":1}])
        data: Dict[str, Any] = {}

        # 先识别 items=...
        items_param = kw.get("items", None)
        if len(pos) >= 1 and isinstance(pos[0], (list, tuple)) and all(isinstance(e, dict) for e in pos[0]):
            items_param = pos[0]

        if items_param is not None:
            if not isinstance(items_param, (list, tuple)):
                raise ValueError("BUY(items=...) must be a list of dicts.")
            normalized: List[Dict[str, Any]] = []
            for e in items_param:
                if not isinstance(e, dict):
                    raise ValueError("Each element in BUY(items=...) must be a dict.")
                iid = e.get("item_id") or e.get("name") or e.get("item")
                try:
                    q = int(e.get("qty", 1))
                except Exception:
                    q = 0
                if iid and q > 0:
                    normalized.append({"item_id": str(iid), "qty": q})
            if not normalized:
                raise ValueError("BUY(items=...) has no valid entries.")
            data["items"] = normalized

        # 再看是否还有单品参数（与 items 同时出现则合并）
        single_item = None
        single_qty = 1
        if len(pos) >= 1 and isinstance(pos[0], str):
            single_item = pos[0]
            if len(pos) >= 2:
                try:
                    single_qty = int(pos[1])
                except Exception:
                    single_qty = 1
        else:
            single_item = kw.get("item") or kw.get("item_id") or kw.get("name")
            try:
                single_qty = int(kw.get("qty", 1))
            except Exception:
                single_qty = 1

        if single_item:
            if single_qty <= 0:
                raise ValueError("BUY qty must be > 0.")
            if "items" in data:
                data["items"].append({"item_id": str(single_item), "qty": int(single_qty)})
            else:
                data.update(dict(item_id=str(single_item), qty=int(single_qty)))

        if not data:
            raise ValueError("BUY needs item_id/name + qty, or items=[{item_id/name, qty}, ...].")

        return DMAction(DMActionKind.BUY, data=data)

    if name == "USE_BATTERY_PACK":
        return DMAction(DMActionKind.USE_BATTERY_PACK, data=dict(item_id="escooter_battery_pack"))

    if name == "USE_ENERGY_DRINK":
        return DMAction(DMActionKind.USE_ENERGY_DRINK, data=dict(item_id="energy_drink"))

    if name == "SWITCH":
        to = (str(pos[0]) if len(pos) >= 1 else str(kw.get("to") or "")).strip()
        if not to:
            raise ValueError("SWITCH needs target 'to'.")
        return DMAction(DMActionKind.SWITCH_TRANSPORT, data=dict(to=to))

    if name == "RENT_CAR":
        rate = float(kw.get("rate_per_min", 1.0))
        speed_m_s = float(kw.get("avg_speed_m_s", 20.0))
        if len(pos) >= 1:
            rate = float(pos[0])
        if len(pos) >= 2:
            speed_m_s = float(pos[1])
        speed_cm_s = float(speed_m_s) * 100.0
        return DMAction(DMActionKind.RENT_CAR, data=dict(rate_per_min=rate, avg_speed_cm_s=speed_cm_s))

    if name == "RETURN_CAR":
        return DMAction(DMActionKind.RETURN_CAR, data={})

    if name == "VIEW_HELP_BOARD":
        return DMAction(DMActionKind.VIEW_HELP_BOARD, data={})

    if name == "POST_HELP":
        kind = str(kw.get("kind") or (pos[0] if len(pos) >= 1 else "")).strip().upper()
        if not kind:
            raise ValueError("POST_HELP needs kind.")
        bounty = float(kw.get("bounty", 0.0))
        ttl_s = float(kw.get("ttl_s", 0.0))
        payload = kw.get("payload", {})
        if len(pos) >= 2 and isinstance(pos[1], (int, float)): bounty = float(pos[1])
        if len(pos) >= 3 and isinstance(pos[2], (int, float)): ttl_s = float(pos[2])
        if len(pos) >= 4 and isinstance(pos[3], (dict, list, tuple, str)):
            payload = pos[3]
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    payload = {"raw": payload}
        return DMAction(DMActionKind.POST_HELP_REQUEST, data=dict(help_type=kind, bounty=bounty, ttl_s=ttl_s, payload=payload))

    if name == "ACCEPT_HELP":
        req_id = int(pos[0]) if len(pos) >= 1 else int(kw.get("req_id"))
        return DMAction(DMActionKind.ACCEPT_HELP_REQUEST, data=dict(req_id=req_id))

    if name == "EDIT_HELP":
        req_id = int(kw.get("req_id", pos[0] if len(pos) >= 1 else -1))
        if req_id == -1:
            raise ValueError("EDIT_HELP needs req_id.")
        data: Dict[str, Any] = dict(req_id=req_id)
        if len(pos) >= 2:
            data["new_bounty"] = float(pos[1])
        if "new_bounty" in kw:
            data["new_bounty"] = float(kw["new_bounty"])
        ttl_min: Optional[float] = None
        if len(pos) >= 3:
            ttl_min = float(pos[2])
        if "new_ttl_min" in kw:
            ttl_min = float(kw["new_ttl_min"])
        if ttl_min is not None:
            data["new_ttl_s"] = float(ttl_min) * 60.0
        return DMAction(DMActionKind.EDIT_HELP_REQUEST, data=data)

    if name == "PLACE_TEMP_BOX":
        req_id = int(kw.get("req_id", pos[0] if len(pos) >= 1 else -1))
        if req_id == -1:
            raise ValueError("PLACE_TEMP_BOX needs req_id.")
        location = kw.get("location", None)
        if len(pos) >= 2 and isinstance(pos[1], (list, tuple)):
            location = tuple(pos[1])
        content = kw.get("content", {})
        if len(pos) >= 3 and isinstance(pos[2], (dict, str)):
            content = pos[2]
        if isinstance(content, str):
            content = json.loads(content)
        data = dict(req_id=req_id, location_xy=tuple(location) if location else None, content=content)
        return DMAction(DMActionKind.PLACE_TEMP_BOX, data=data)

    if name == "TAKE_FROM_TEMP_BOX":
        req_id = int(kw.get("req_id", pos[0] if len(pos) >= 1 else -1))
        if req_id == -1:
            raise ValueError("TAKE_FROM_TEMP_BOX needs req_id.")
        return DMAction(DMActionKind.TAKE_FROM_TEMP_BOX, data=dict(req_id=req_id))

    if name == "REPORT_HELP_FINISHED":
        req_id = int(kw.get("req_id", pos[0] if len(pos) >= 1 else -1))
        if req_id == -1:
            raise ValueError("REPORT_HELP_FINISHED needs req_id.")
        return DMAction(DMActionKind.REPORT_HELP_FINISHED, data=dict(req_id=req_id))

    # --- NEW: DROP_OFF(oid=..., method=...)
    if name == "DROP_OFF":
        # 位置参数形态：DROP_OFF(12, "leave_at_door")
        # 关键字形态：   DROP_OFF(oid=12, method="knock")
        if len(pos) >= 1 and isinstance(pos[0], (int, float)):
            oid = int(pos[0])
        else:
            oid = int(kw.get("oid"))
        method = None
        if len(pos) >= 2 and isinstance(pos[1], str):
            method = pos[1]
        else:
            method = kw.get("method")
        if method is None:
            raise ValueError("DROP_OFF needs 'method'.")
        method_norm = str(method).strip().lower().replace("-", "_").replace(" ", "_")
        if method_norm not in {"leave_at_door", "knock", "call", "hand_to_customer"}:
            raise ValueError("DROP_OFF.method must be one of leave_at_door|knock|call|hand_to_customer")
        return DMAction(DMActionKind.DROP_OFF, data={"oid": int(oid), "method": method_norm})

    if name == "VIEW_BAG":
        return DMAction(DMActionKind.VIEW_BAG, data={})

    if name == "USE_ICE_PACK":
        # 接受位置/关键字：USE_ICE_PACK("A") / USE_ICE_PACK(comp="A")
        raw = (pos[0] if len(pos) >= 1 else kw.get("comp"))
        s = (str(raw).strip().upper() if raw is not None else "")
        if not s or not ('A' <= s[0] <= 'Z'):
            raise ValueError("USE_ICE_PACK needs comp like 'A'/'B'/...")
        return DMAction(DMActionKind.USE_ICE_PACK, data=dict(comp=s[0]))

    if name == "USE_HEAT_PACK":
        raw = (pos[0] if len(pos) >= 1 else kw.get("comp"))
        s = (str(raw).strip().upper() if raw is not None else "")
        if not s or not ('A' <= s[0] <= 'Z'):
            raise ValueError("USE_HEAT_PACK needs comp like 'A'/'B'/...")
        return DMAction(DMActionKind.USE_HEAT_PACK, data=dict(comp=s[0]))

    if name == "SAY":
        # 允许：
        #  - SAY("hello")                      -> 广播
        #  - SAY(text="hello")                 -> 广播
        #  - SAY("hello", to="7")              -> 私聊
        #  - SAY(to="7", text="hello")         -> 私聊
        #  - SAY(text="hi", to="ALL"|"*")      -> 广播
        if len(pos) >= 1 and isinstance(pos[0], str):
            text = pos[0]
        else:
            text = kw.get("text") or ""

        to = kw.get("to", None)
        # 兼容数字ID：SAY(to=7, text="...") -> "7"
        if isinstance(to, (int, float)):
            to = str(int(to))
        if to is not None:
            t = str(to).strip()
            # 标准化广播标记
            if t == "" or t.upper() in ("ALL", "*"):
                to = None
            else:
                to = t

        text = str(text).strip()
        if not text:
            raise ValueError("SAY needs non-empty 'text' (e.g., SAY(\"hello\") or SAY(text=\"hello\")).")

        return DMAction(DMActionKind.SAY, data={"text": text, "to": to})



    raise ValueError(f"Unrecognized or malformed action: {str(model_text)[:120]}")

# =========================================================
# 4) 文本化（英文简述）
# =========================================================
def _fmt_m(cm: Optional[float]) -> str:
    if cm is None:
        return "N/A"
    return f"{(float(cm) / 100.0):.2f} m"

def _fmt_xy_cm(xy: Optional[Tuple[float, float]]) -> str:
    if not xy:
        return "(N/A, N/A)"
    x, y = float(xy[0]), float(xy[1])
    return f"({x/100.0:.2f} m, {y/100.0:.2f} m)"

def _order_brief(o: Any) -> str:
    oid = getattr(o, "id", None)
    pu = getattr(o, "pickup_road_name", None) or ""
    do = getattr(o, "dropoff_road_name", None) or ""
    core = f"#{oid}" if oid is not None else "(unknown)"
    if pu or do:
        return f"order {core} ({pu} → {do})"
    return f"order {core}"

def action_to_text(action: Any, dm: Optional[Any] = None) -> str:
    """
    把 DMAction 转为简洁英文描述（含关键信息）。
    - 不抛异常，尽量用可得信息组织短句。
    """
    kind = getattr(action, "kind", None)
    data: Dict[str, Any] = getattr(action, "data", {}) or {}

    # 为特定动作准备上下文工具
    def _order_ids_text(orders: List[Any]) -> str:
        ids = [getattr(o, "id", None) for o in orders if getattr(o, "id", None) is not None]
        return ", ".join(f"#{i}" for i in ids) if ids else "(none)"

    # MOVE_TO
    if str(kind).endswith("MOVE_TO"):
        tx = data.get("tx"); ty = data.get("ty")
        note = ""
        if "expected_dist_cm" in data:
            note = f" (~{float(data['expected_dist_cm'])/100.0:.2f} m expected)"
        return f"Move to {_fmt_xy_cm((tx, ty))}{note}."

    # VIEW_ORDERS
    if str(kind).endswith("VIEW_ORDERS"):
        return "View available orders."

    # ACCEPT_ORDER
    if str(kind).endswith("ACCEPT_ORDER"):
        data = data or {}
        if "oids" in data and data["oids"]:
            ids = ", ".join(f"#{int(i)}" for i in data["oids"])
            return f"Accept orders {ids}."
        if "oid" in data:
            return f"Accept order #{int(data['oid'])}."
        o = data.get("order")  # 兼容旧写法
        if o is not None:
            return f"Accept {_order_brief(o)}."
        return "Accept the order."

    # PICKUP
    if str(kind).endswith("PICKUP"):
        orders = list(data.get("orders") or [])
        if orders:
            place = getattr(orders[0], "pickup_road_name", None)
            suffix = f" at {place}" if place else ""
            return f"Pick up orders {_order_ids_text(orders)}{suffix}."
        return "Pick up ready orders."

    # PLACE_FOOD_IN_BAG
    if str(kind).endswith("PLACE_FOOD_IN_BAG"):
        cmd = (data.get("bag_cmd") or "").strip()
        return f'Place items into the insulated bag using: "{cmd}".' if cmd else "Place items into the insulated bag."

    # CHARGE_ESCOOTER
    if str(kind).endswith("CHARGE_ESCOOTER"):
        tgt = data.get("target_pct")
        return f"Charge the e-scooter to {float(tgt):.0f}%." if tgt is not None else "Charge the e-scooter."

    # WAIT
    if str(kind).endswith("WAIT"):
        if data.get("until") == "charge_done":
            return "Wait until charging completes."
        secs = data.get("duration_s")
        return f"Wait for {int(secs)} seconds." if secs else "Wait."

    # REST
    if str(kind).endswith("REST"):
        tgt = data.get("target_pct")
        return f"Rest until energy reaches {float(tgt):.0f}%." if tgt is not None else "Rest."

    # BUY
    if str(kind).endswith("BUY"):
        if "items" in data and data["items"]:
            parts = []
            for it in data["items"]:
                iid = it.get("item_id") or it.get("name") or "item"
                q = int(it.get("qty", 1))
                parts.append(f"{q} × {iid}")
            return "Buy " + ", ".join(parts) + "."
        item = data.get("item_id") or data.get("name") or "item"
        qty = int(data.get("qty", 1))
        return f"Buy {qty} × {item}."

    # USE_BATTERY_PACK / USE_ENERGY_DRINK
    if str(kind).endswith("USE_BATTERY_PACK"):
        return "Use a scooter battery pack (recharge to 100%)."
    if str(kind).endswith("USE_ENERGY_DRINK"):
        return "Use an energy drink (+50% energy)."

    # SWITCH_TRANSPORT
    if str(kind).endswith("SWITCH_TRANSPORT"):
        to = str(data.get("to") or "").strip() or "target mode"
        return f"Switch to {to}."

    # RENT_CAR / RETURN_CAR
    if str(kind).endswith("RENT_CAR"):
        rate = float(data.get("rate_per_min", 0.0))
        spd_cm_s = float(data.get("avg_speed_cm_s", 0.0))
        return f"Rent a car @ ${rate:.2f}/min (avg ~{spd_cm_s/100.0:.1f} m/s)."
    if str(kind).endswith("RETURN_CAR"):
        return "Return the rental car."

    # VIEW_HELP_BOARD
    if str(kind).endswith("VIEW_HELP_BOARD"):
        return "View help board."

    # POST_HELP_REQUEST
    if str(kind).endswith("POST_HELP_REQUEST"):
        kind_s = str(data.get("help_type") or "").upper()
        bounty = float(data.get("bounty", 0.0))
        ttl_s = float(data.get("ttl_s", 0.0))
        payload = dict(data.get("payload") or {})
        mins = int(round(ttl_s / 60.0)) if ttl_s else 0

        def _fmt_payload():
            if kind_s == "HELP_DELIVERY":
                oid = payload.get("order_id")
                provide_xy = payload.get("provide_xy")
                brief = ""
                if dm is not None and oid is not None:
                    o = _lookup_order(dm, int(oid))
                    if o is not None:
                        brief = f" {_order_brief(o)}"
                return f"handoff at {_fmt_xy_cm(provide_xy)} for order #{oid}{brief}"
            if kind_s == "HELP_BUY":
                inv = payload.get("buy_list") or {}
                if isinstance(inv, list):
                    pairs = inv
                else:
                    pairs = list(inv.items())
                parts = [f"{k}×{v}" for (k, v) in pairs]
                deliver_xy = payload.get("deliver_xy")
                return f"shopping list [{', '.join(parts)}], deliver to {_fmt_xy_cm(deliver_xy)}"
            if kind_s == "HELP_CHARGE":
                provide_xy = payload.get("provide_xy")
                deliver_xy = payload.get("deliver_xy")
                target = payload.get("want_charge_pct", payload.get("target_pct", 100.0))
                return f"pick at {_fmt_xy_cm(provide_xy)}, charge to {float(target):.0f}%, deliver to {_fmt_xy_cm(deliver_xy)}"
            return "details provided"

        return f"Post help request ({kind_s}), bounty ${bounty:.2f}, TTL {mins} min: {_fmt_payload()}."

    # ACCEPT_HELP_REQUEST
    if str(kind).endswith("ACCEPT_HELP_REQUEST"):
        req_id = data.get("req_id")
        desc = f"Accept help request #{req_id}"
        try:
            from Base.Comms import get_comms
            comms = get_comms()
            if comms and req_id is not None:
                det = comms.get_request_detail(req_id=int(req_id))
                if det:
                    who = det.get("publisher_id")
                    k = det.get("kind")
                    if who or k:
                        extra = []
                        if k: extra.append(str(k))
                        if who: extra.append(f"from {who}")
                        desc += f" ({', '.join(extra)})"
        except Exception:
            pass
        return desc + "."

    # EDIT_HELP_REQUEST
    if str(kind).endswith("EDIT_HELP_REQUEST"):
        req_id = data.get("req_id")
        parts = []
        if "new_bounty" in data:
            parts.append(f"bounty -> ${float(data['new_bounty']):.2f}")
        if "new_ttl_s" in data:
            parts.append(f"TTL -> {int(round(float(data['new_ttl_s'])/60.0))} min")
        tail = f" ({', '.join(parts)})" if parts else ""
        return f"Edit help request #{req_id}{tail}."

    # PLACE_TEMP_BOX
    if str(kind).endswith("PLACE_TEMP_BOX"):
        req_id = data.get("req_id")
        loc = data.get("location_xy")
        content = dict(data.get("content") or {})
        parts = []
        inv = content.get("inventory") or {}
        if inv:
            inv_s = ", ".join(f"{k}×{v}" for k, v in inv.items())
            parts.append(f"inventory[{inv_s}]")
        if "food" in content:
            parts.append("all food items")
        if "escooter" in content:
            parts.append("e-scooter")
        stuff = "; ".join(parts) if parts else "items"
        loc_s = _fmt_xy_cm(tuple(loc)) if loc else "current location"
        return f"Place a temp box for request #{req_id} at {loc_s} containing {stuff}."

    # TAKE_FROM_TEMP_BOX
    if str(kind).endswith("TAKE_FROM_TEMP_BOX"):
        req_id = data.get("req_id")
        return f"Take items from the temp box for request #{req_id}."

    # REPORT_HELP_FINISHED
    if str(kind).endswith("REPORT_HELP_FINISHED"):
        req_id = data.get("req_id")
        return f"Report help finished for request #{req_id}."

    # DROP_OFF
    if str(kind).endswith("DROP_OFF"):
        oid = data.get("oid")
        method = data.get("method")
        return f"Drop off order #{oid} via {method}."

    # VIEW_BAG
    if str(kind).endswith("VIEW_BAG"):
        return "View insulated bag layout."

    # USE_ICE_PACK / USE_HEAT_PACK
    if str(kind).endswith("USE_ICE_PACK"):
        c = (data.get("comp") or "?")
        return f"Use an ice pack on compartment {c}."
    if str(kind).endswith("USE_HEAT_PACK"):
        c = (data.get("comp") or "?")
        return f"Use a heat pack on compartment {c}."

    if str(kind).endswith("SAY"):
        txt = str(data.get("text") or "")
        to  = data.get("to", None)
        # 为了简洁，长文本裁剪一下（可选）
        show = (txt if len(txt) <= 80 else (txt[:77] + "..."))
        if to:
            return f'Send chat to {to}: "{show}"'
        return f'Send broadcast chat: "{show}"'

    # Fallback
    return "Execute action."