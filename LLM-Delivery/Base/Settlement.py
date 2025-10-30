# -*- coding: utf-8 -*-
# Base/Settlement.py
"""
Order settlement system.

- 基础工资 + 奖惩
- 星级（0~5）：时间星、食物星（温度/串味/破坏）、方式星
- 统计指标：是否按时、温度是否合规、是否串味、是否破坏
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Dict
import math

# ---------------- Config ----------------

@dataclass
class SettlementConfig:
    # 奖惩（美元）
    bonus_5_star: float = 3.0
    bonus_4_star: float = 1.5
    penalty_1_star: float = -2.0
    penalty_0_star: float = -2.0

    # 超时基础工资折减策略
    enable_base_overtime_penalty: bool = True
    overtime_grace_s: float = 60.0
    overtime_linear_min_base_fraction: float = 0.50

    # 时间星级阈值（相对时限倍率）
    time_star_cutoffs: tuple[float, float, float, float] = (1.00, 1.10, 1.25, 1.50)

    # 食物温度阈值（绝对偏差，°C）：|temp - serving|
    temp_delta_cutoffs_c: tuple[float, float, float, float] = (4.0, 7.0, 10.0, 15.0)

    # ===== 破坏等级（0~3）映射至星级 =====
    # 索引即 damage_level：0→5★，1→3★，2→2★，3→1★
    damage_level_to_star: tuple[int, int, int, int] = (5, 3, 2, 1)

    # ===== 串味（odor_contamination ∈ [0,1]，越小越好）映射至星级 =====
    # 用绝对量分档（小于等于 c1 计 5★，>c4 计 1★）
    odor_contam_cutoffs: tuple[float, float, float, float] = (0.05, 0.15, 0.30, 0.50)

    # ===== 食物质量权重（建议和=1.0）=====
    food_temp_weight: float = 0.5
    food_odor_weight: float = 0.2
    food_damage_weight: float = 0.3

    # ===== 统计判定阈值（布尔指标用）=====
    temp_ok_max_delta_c: float = 7.0          # 温度“正常范围”的最大偏差
    odor_ok_threshold: float = 0.30           # 串味“合格阈值”（≤ 视为 OK）
    damage_fail_level: int = 2                # 破坏等级达到/超过该值视为“破坏”

# ---------------- Result ----------------

@dataclass
class SettlementResult:
    stars: int
    base_pay: float
    extra_pay: float
    total_pay: float
    breakdown: Dict[str, Any] = field(default_factory=dict)

# ---------------- Helpers ----------------

def _star_from_ratio(ratio: float, cutoffs: tuple[float, float, float, float]) -> int:
    c1, c2, c3, c4 = cutoffs
    if ratio <= c1: return 5
    if ratio <= c2: return 4
    if ratio <= c3: return 3
    if ratio <= c4: return 2
    return 1

def _star_from_abs(x: float, cutoffs: tuple[float, float, float, float], reverse: bool = False) -> int:
    c1, c2, c3, c4 = cutoffs
    if not reverse:
        if x <= c1: return 5
        if x <= c2: return 4
        if x <= c3: return 3
        if x <= c4: return 2
        return 1
    if x >= c1: return 5
    if x >= c2: return 4
    if x >= c3: return 3
    if x >= c4: return 2
    return 1

def _safe_float(v: Any, default: float = math.nan) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)

def _safe_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(v) if v is not None else default
    except Exception:
        return default

# ---------------- Public API ----------------

def compute_settlement(
    *,
    order_base_earnings: float,
    duration_s: float,
    time_limit_s: float,
    items: Iterable[Any],
    config: Optional[SettlementConfig] = None,
    order_allowed_delivery_methods: Optional[Iterable[str]] = None,
    actual_delivery_method: Optional[str] = None,
) -> SettlementResult:
    """
    计算一次订单送达的结算结果。
    食物质量仅基于：温度偏差、串味（odor_contamination）、破坏等级（0~3）。
    'strong' 气味的食物不参与串味惩罚（按 5★ 处理）。
    """
    cfg = config or SettlementConfig()

    # ---------- Base pay ----------
    base_pay = float(order_base_earnings)
    base_note = "on time"
    overtime_s = 0.0
    on_time = True
    if time_limit_s and duration_s:
        if duration_s <= time_limit_s + cfg.overtime_grace_s:
            base_pay = order_base_earnings
            base_note = "on time"
            on_time = True
        else:
            on_time = False
            overtime_s = duration_s - time_limit_s
            if cfg.enable_base_overtime_penalty:
                ratio = duration_s / max(1e-9, time_limit_s)
                fraction = max(cfg.overtime_linear_min_base_fraction, 1.0 / ratio)
                base_pay = order_base_earnings * fraction
                base_note = f"overtime ratio={ratio:.2f}, fraction={fraction:.2f}"
            else:
                base_pay = order_base_earnings
                base_note = "overtime (no base penalty)"

    # ---------- Time star ----------
    if time_limit_s > 0:
        time_ratio = duration_s / max(1e-9, time_limit_s)
        time_star = _star_from_ratio(time_ratio, cfg.time_star_cutoffs)
    else:
        time_star = 5

    # ---------- Food star（温度 + 串味 + 破坏） ----------
    per_item_stars: list[int] = []

    per_item_temp_delta: list[float] = []
    per_item_temp_star: list[Optional[int]] = []
    per_item_temp_ok: list[Optional[bool]] = []

    per_item_odor: list[Optional[str]] = []
    per_item_odor_contam: list[float] = []
    per_item_odor_star: list[Optional[int]] = []
    per_item_odor_ok: list[Optional[bool]] = []

    per_item_damage_level: list[Optional[int]] = []
    per_item_damage_star: list[Optional[int]] = []
    per_item_damage_violation: list[Optional[bool]] = []

    any_temp_issue = False
    any_odor_contamination = False
    any_damage_violation = False

    for it in items or []:
        # 读取温度
        temp_c = _safe_float(getattr(it, "temp_c", None))
        if isinstance(it, dict) and math.isnan(temp_c):
            temp_c = _safe_float(it.get("temp_c", None))
        serving_c = _safe_float(getattr(it, "serving_temp_c", None))
        if isinstance(it, dict) and math.isnan(serving_c):
            serving_c = _safe_float(it.get("serving_temp_c", None))

        # 读取 odor / odor_contamination
        odor = getattr(it, "odor", None)
        if odor is None and isinstance(it, dict):
            odor = it.get("odor", None)
        odor_str = (str(odor).strip().lower() if odor is not None else None)

        oc = _safe_float(getattr(it, "odor_contamination", None))
        if isinstance(it, dict) and math.isnan(oc):
            oc = _safe_float(it.get("odor_contamination", None))
        if not math.isnan(oc):
            oc = max(0.0, min(1.0, oc))

        # 读取 damage_level（0~3）
        dl = _safe_int(getattr(it, "damage_level", None))
        if dl is None and isinstance(it, dict):
            dl = _safe_int(it.get("damage_level", None))
        if dl is not None:
            dl = max(0, min(3, dl))

        # ---- 子星级：温度 ----
        t_star: Optional[int] = None
        t_delta = math.nan
        t_ok: Optional[bool] = None
        if not math.isnan(temp_c) and not math.isnan(serving_c):
            t_delta = abs(temp_c - serving_c)
            t_star = _star_from_abs(t_delta, cfg.temp_delta_cutoffs_c)
            t_ok = (t_delta <= cfg.temp_ok_max_delta_c)
            if t_ok is False:
                any_temp_issue = True

        # ---- 子星级：串味 ----
        o_star: Optional[int] = None
        o_ok: Optional[bool] = None
        if odor_str == "strong":
            # 本身重味，不惩罚；按 5★，也视为合格
            o_star = 5
            o_ok = True
        else:
            if not math.isnan(oc):
                o_star = _star_from_abs(oc, cfg.odor_contam_cutoffs)
                o_ok = (oc <= cfg.odor_ok_threshold)
                if o_ok is False:
                    any_odor_contamination = True
            # 若没有 oc 数据，保持 None（该维度缺失）

        # ---- 子星级：破坏 ----
        d_star: Optional[int] = None
        d_violate: Optional[bool] = None
        if dl is not None:
            d_star = int(cfg.damage_level_to_star[dl])
            d_violate = (dl >= cfg.damage_fail_level)
            if d_violate:
                any_damage_violation = True

        # ---- 合成星（仅基于存在的信息；均缺则 3★ 兜底）----
        # 使用三个权重：temp/odor/damage
        components = []
        weights = []

        if t_star is not None:
            components.append(t_star); weights.append(cfg.food_temp_weight)
        if o_star is not None:
            components.append(o_star); weights.append(cfg.food_odor_weight)
        if d_star is not None:
            components.append(d_star); weights.append(cfg.food_damage_weight)

        if not components:
            s = 3
        else:
            # 仅对存在的维度做加权；若仅有 1~2 个维度，则按对应权重求和后再按“使用到的权重之和”归一
            w_sum = sum(weights) if weights else 1.0
            s_float = sum(c * w for c, w in zip(components, weights)) / max(1e-9, w_sum)
            s = int(round(s_float))
        s = max(1, min(5, s))
        per_item_stars.append(s)

        # 记录明细
        per_item_temp_delta.append(t_delta)
        per_item_temp_star.append(t_star)
        per_item_temp_ok.append(t_ok)

        per_item_odor.append(odor_str)
        per_item_odor_contam.append(oc)
        per_item_odor_star.append(o_star)
        per_item_odor_ok.append(o_ok)

        per_item_damage_level.append(dl)
        per_item_damage_star.append(d_star)
        per_item_damage_violation.append(d_violate)

    if per_item_stars:
        food_star = int(round(sum(per_item_stars) / len(per_item_stars)))
        food_star = max(1, min(5, food_star))
    else:
        food_star = 3

    # ---------- Method star ----------
    allowed_set = {str(m).strip().lower() for m in (order_allowed_delivery_methods or []) if str(m).strip()}
    actual = (actual_delivery_method or "").strip().lower()
    if not allowed_set:
        method_star = 5
        method_note = "no_preference"
    else:
        if actual and (actual in allowed_set):
            method_star = 5
            method_note = "match"
        else:
            method_star = 1
            method_note = "mismatch_or_unknown"

    # ---------- Overall stars ----------
    stars = int(round((time_star + food_star + method_star) / 3.0))
    stars = max(0, min(5, stars))

    # ---------- Extra pay ----------
    if stars >= 5:
        extra = cfg.bonus_5_star
        extra_note = "5★ bonus"
    elif stars == 4:
        extra = cfg.bonus_4_star
        extra_note = "4★ bonus"
    elif stars == 3:
        extra = 0.0
        extra_note = "3★ no bonus"
    elif stars == 2:
        extra = 0.0
        extra_note = "2★ no bonus"
    elif stars == 1:
        extra = cfg.penalty_1_star
        extra_note = "1★ penalty"
    else:
        extra = cfg.penalty_0_star
        extra_note = "0★ penalty"

    total = float(base_pay) + float(extra)

    # ---------- 统计布尔指标（用于 failure cases 汇总） ----------
    # 若无相关数据，则以下 *_all 默认 True（表示未观测到问题）
    temp_ok_all = (not any_temp_issue)
    odor_ok_all = (not any_odor_contamination)
    damage_ok_all = (not any_damage_violation)

    return SettlementResult(
        stars=stars,
        base_pay=float(base_pay),
        extra_pay=float(extra),
        total_pay=float(total),
        breakdown=dict(
            flags=dict(  # 关键统计布尔指标
                on_time=bool(on_time),
                temp_ok_all=bool(temp_ok_all),
                has_temp_issue=bool(any_temp_issue),
                odor_ok_all=bool(odor_ok_all),
                has_odor_contamination=bool(any_odor_contamination),
                damage_ok_all=bool(damage_ok_all),
                has_damage_violation=bool(any_damage_violation),
            ),
            time=dict(
                duration_s=float(duration_s),
                time_limit_s=float(time_limit_s),
                overtime_s=float(overtime_s),
                time_star=int(time_star),
                note=base_note
            ),
            food=dict(
                per_item_stars=list(per_item_stars),
                food_star=int(food_star),
                weights=dict(
                    temp=float(cfg.food_temp_weight),
                    odor=float(cfg.food_odor_weight),
                    damage=float(cfg.food_damage_weight),
                ),
                temp=dict(
                    cutoffs_c=tuple(cfg.temp_delta_cutoffs_c),
                    ok_max_delta_c=float(cfg.temp_ok_max_delta_c),
                    per_item_delta=list(per_item_temp_delta),
                    per_item_star=list(per_item_temp_star),
                    per_item_ok=list(per_item_temp_ok),
                ),
                odor=dict(
                    cutoffs=tuple(cfg.odor_contam_cutoffs),
                    ok_threshold=float(cfg.odor_ok_threshold),
                    per_item_odor=list(per_item_odor),
                    per_item_contam=list(per_item_odor_contam),
                    per_item_star=list(per_item_odor_star),
                    per_item_ok=list(per_item_odor_ok),
                ),
                damage=dict(
                    level_to_star=tuple(cfg.damage_level_to_star),
                    fail_level=int(cfg.damage_fail_level),
                    per_item_level=list(per_item_damage_level),
                    per_item_star=list(per_item_damage_star),
                    per_item_violation=list(per_item_damage_violation),
                ),
            ),
            method=dict(
                allowed=sorted(list(allowed_set)),
                actual=actual,
                method_star=int(method_star),
                note=method_note
            ),
            policy=dict(config=cfg.__dict__),
            extra_note=extra_note
        )
    )