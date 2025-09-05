# -*- coding: utf-8 -*-
"""
ActionSpace.py
- 提供 ACTION_API_SPEC：动作空间/输出格式的固定说明文本（不会替换你的 vlm_prompt）
- 提供 parse_action(...)：把 VLM 一行输出解析为 DMAction（与 Base/DeliveryMan.py 里的 handlers 一一对应）
- 解析健壮：去围栏、大小写、位置/命名参数、少量同义命令、字面量 AST 安全解析

改动（统一新格式，不考虑老格式）：
- MOVE(x, y) 使用米单位，必须带 m 后缀，例如 MOVE(116.24m, -224.26m)。解析时统一换算为厘米。
- RENT_CAR(rate_per_min=..., avg_speed_m_s=...) 速度采用 m/s；解析后换算为 cm/s 传给引擎。
- EDIT_HELP(req_id=..., new_bounty=..., new_ttl_min=...) duration 用分钟；解析后换算为秒传给引擎。
"""

from __future__ import annotations
import ast
import json
import re
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["ACTION_API_SPEC", "parse_action", "sanitize_model_text"]

# =========================================================
# 1) 只描述“动作空间+输出规则”的规范文本（非 system prompt）
# =========================================================
ACTION_API_SPEC: str = r"""
You must output EXACTLY ONE action per turn, in a single-line function-call format.
No prose, no code fences, no comments. Pick one command below.

COMMANDS (UPPERCASE):
- MOVE(x, y)                          # x,y in meters, MUST use the 'm' suffix (e.g., 102.3m, -5.0m)
- VIEW_ORDERS()
- ACCEPT_ORDER(order_id)
- PICKUP(bag_cmd="order <id>: 1,2 -> A; 3 -> B")
- CHARGE(target_pct=100)
- REST(target_pct=100)
- WAIT(seconds=NN) or WAIT("charge_done")
- BUY(item="energy_drink", qty=1)
- USE_BATTERY_PACK()
- USE_ENERGY_DRINK()
- SWITCH(to="walk"|"e-scooter"|"car")
- RENT_CAR(rate_per_min=1.0, avg_speed_m_s=20)        # speed in m/s
- RETURN_CAR()
- POST_HELP(kind="HELP_PICKUP"|"HELP_DELIVERY"|"HELP_BUY"|"HELP_CHARGE", bounty=5.0, ttl_s=300, payload={...})
- ACCEPT_HELP(req_id=123)
- EDIT_HELP(req_id=123, new_bounty=6.0, new_ttl_min=10)  # duration in minutes

PRECONDITIONS (obey strictly):
- PICKUP only when already at the pickup door.
- CHARGE only at a charging_station; scooter may be parked automatically.
- REST only at a rest_area.
- SWITCH to "e-scooter"/"car" only when near your vehicle.
- For deliveries, MOVE to the dropoff; the simulator auto-delivers on arrival.

OUTPUT EXAMPLES (exactly one line):
  MOVE(102.3m, 885.5m)
  VIEW_ORDERS()
  ACCEPT_ORDER(12)
  PICKUP(bag_cmd="order 12: 1,2 -> A; 3 -> B")
  CHARGE(target_pct=100)
  WAIT("charge_done")
  BUY(item="energy_drink", qty=1)
  SWITCH(to="e-scooter")
  RENT_CAR(rate_per_min=1.0, avg_speed_m_s=22.0)
  POST_HELP(kind="HELP_PICKUP", bounty=6.0, ttl_s=600, payload={"order_id": 18})
  EDIT_HELP(req_id=123, new_bounty=7.0, new_ttl_min=8)
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

    "POST_HELP": "POST_HELP",
    "POST_HELP_REQUEST": "POST_HELP",

    "ACCEPT_HELP": "ACCEPT_HELP",
    "ACCEPT_HELP_REQUEST": "ACCEPT_HELP",

    "EDIT_HELP": "EDIT_HELP",
    "EDIT_HELP_REQUEST": "EDIT_HELP",
}

def sanitize_model_text(text: str) -> str:
    """去掉围栏/多余行，只保留形如 NAME(... ) 的首行。"""
    t = (text or "").strip()
    m = FENCE_RE.search(t)
    if m:
        t = m.group(1).strip()
    t = t.splitlines()[0].strip()
    t = t.rstrip(";，。.")
    return t

# —— 将不在字符串内的 “<number>m” 替换为换算到厘米后的纯数字（字符串内部保持原样）
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
        # 不在字符串时，尝试匹配 “<number>m”
        m = _UNIT_NUM_M.match(s, i)
        if m:
            num_m = float(m.group('num'))
            num_cm = num_m * 100.0
            # 格式化为尽可能简洁的数字文本
            if abs(num_cm - round(num_cm)) < 1e-9:
                out.append(str(int(round(num_cm))))
            else:
                out.append(f"{num_cm:.6f}".rstrip('0').rstrip('.'))
            i = m.end()
            continue
        # 进入字符串
        if ch in ('"', "'"):
            in_str, quote = True, ch
        out.append(ch)
        i += 1
    return ''.join(out)

def _ast_value(node: ast.AST) -> Any:
    """把 AST 字面量安全转成 Python 值（常量/字典/列表/元组/True/False/None/字符串拼接）。"""
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
    """把 'NAME(arg1, k=v, ...)' 解析为 (NAME, [pos...], {kw...})。"""
    m = _CALL_RE.match(text)
    if not m:
        raise ValueError("Output not a function call like NAME(...).")
    raw_name = m.group(1).strip()
    name = _CANON.get(raw_name.upper(), raw_name.upper())
    args_src = m.group(2).strip()
    if args_src == "":
        return name, [], {}

    # 对参数部分进行单位换算：
    # - 坐标中的 “<number>m” 统一转为厘米（纯数字），便于 AST 解析与内部使用
    args_src = _convert_m_to_cm_in_args(args_src)

    fake = f"__F__({args_src})"
    tree = ast.parse(fake, mode="eval")
    if not isinstance(tree.body, ast.Call):
        raise ValueError("Not a call.")
    call = tree.body
    pos = [_ast_value(a) for a in call.args]
    kw = {kw.arg: _ast_value(kw.value) for kw in call.keywords if kw.arg is not None}
    return name, pos, kw

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
    if om is not None:
        for method in ("get_by_id", "get_order", "find", "get"):
            if hasattr(om, method):
                o = getattr(om, method)(oid)  # type: ignore
                if o is not None:
                    return o
        pool = list(getattr(om, "_orders", []) or [])
        for o in pool:
            if getattr(o, "id", None) == oid:
                return o
    for o in getattr(dm, "active_orders", []) or []:
        if getattr(o, "id", None) == oid:
            return o
    return None

# =========================================================
# 3) 入口：解析为 DMAction
# =========================================================
def parse_action(model_text: str, dm: Any):
    """
    把 VLM 的一行输出解析为 DMAction。
    成功返回 DMAction；失败抛 ValueError（外层可记录到 dm.vlm_errors）。
    """
    from Base.DeliveryMan import DMAction, DMActionKind  # 延迟导入避免循环

    text = sanitize_model_text(model_text)
    name, pos, kw = _parse_call(text)

    if name == "MOVE":
        # 坐标已在 _parse_call 中完成 m -> cm 的换算
        if len(pos) >= 2:
            x, y = float(pos[0]), float(pos[1])
        else:
            x, y = float(kw.get("x")), float(kw.get("y"))
        return DMAction(DMActionKind.MOVE_TO, data=dict(tx=x, ty=y, use_route=True, snap_cm=120.0))

    if name == "VIEW_ORDERS":
        return DMAction(DMActionKind.VIEW_ORDERS, data={})

    if name == "ACCEPT_ORDER":
        oid = int(pos[0]) if len(pos) >= 1 else int(kw.get("order_id") or kw.get("id"))
        order = _lookup_order(dm, oid)
        if order is None:
            raise ValueError(f"ACCEPT_ORDER: order_id {oid} not found.")
        return DMAction(DMActionKind.ACCEPT_ORDER, data=dict(order=order))

    if name == "PICKUP":
        bag_cmd = (pos[0] if (len(pos) >= 1 and isinstance(pos[0], str)) else kw.get("bag_cmd") or kw.get("cmd") or "").strip()
        if not bag_cmd:
            raise ValueError("PICKUP needs bag_cmd.")
        here_orders = _orders_at_pickup(dm, tol_cm=300.0)
        if not here_orders:
            raise ValueError("PICKUP: no ready orders at current door (MOVE first).")
        return DMAction(DMActionKind.PICKUP, data=dict(orders=here_orders, tol_cm=300.0, bag_cmd=bag_cmd))

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
        if len(pos) >= 1:
            item = str(pos[0]); qty = int(pos[1]) if len(pos) >= 2 else 1
        else:
            item = str(kw.get("item") or kw.get("item_id") or kw.get("name") or "")
            qty = int(kw.get("qty", 1))
        if not item:
            raise ValueError("BUY needs item/name.")
        return DMAction(DMActionKind.BUY, data=dict(item_id=item, qty=qty))

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
        # 速率：avg_speed_m_s（米/秒）→ 换算为 avg_speed_cm_s（厘米/秒）
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

    if name == "POST_HELP":
        kind = str(kw.get("kind") or (pos[0] if len(pos) >= 1 else "")).strip().upper()
        if not kind:
            raise ValueError("POST_HELP needs kind.")
        bounty = float(kw.get("bounty", 0.0))
        ttl_s = float(kw.get("ttl_s", 0.0))  # 这里仍使用秒（保持现有接口不变）
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
        # 现在只支持：req_id、new_bounty（可选）、new_ttl_min（分钟，可选）
        req_id = int(kw.get("req_id", pos[0] if len(pos) >= 1 else -1))
        if req_id == -1:
            raise ValueError("EDIT_HELP needs req_id.")
        data: Dict[str, Any] = dict(req_id=req_id)

        # new_bounty（位置参数或命名参数）
        if len(pos) >= 2:
            data["new_bounty"] = float(pos[1])
        if "new_bounty" in kw:
            data["new_bounty"] = float(kw["new_bounty"])

        # new_ttl_min（位置参数第3个 或 命名参数）
        ttl_min: Optional[float] = None
        if len(pos) >= 3:
            ttl_min = float(pos[2])
        if "new_ttl_min" in kw:
            ttl_min = float(kw["new_ttl_min"])
        if ttl_min is not None:
            data["new_ttl_s"] = float(ttl_min) * 60.0  # 分钟 → 秒

        return DMAction(DMActionKind.EDIT_HELP_REQUEST, data=data)

    raise ValueError(f"Unknown action: {name}")