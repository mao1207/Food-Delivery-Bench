# -*- coding: utf-8 -*-
# Base/Settlement.py
"""
Order settlement system.

功能：
- 将一次送达结算为：基础工资（base） + 额外奖励/扣款（extra）
- 评分为 0~5 星（简单 rule-based，基于花费时间 & 食物状态）
- 在 time limit 内：拿满基础工资（即订单自带 earnings）
- 超时：基础工资按策略折减（默认线性折减，保底 50%）
- 额外奖励：五星/四星加钱；两星/一星/零星扣一点

可通过 SettlementConfig 定制阈值与金额。
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
    # 线性折减：超过 limit 的部分按比例折减，保底基薪比例 >= min_base_fraction
    enable_base_overtime_penalty: bool = True
    overtime_grace_s: float = 60.0           # 宽限（秒），在宽限内不视为超时
    overtime_linear_min_base_fraction: float = 0.50  # 折减保底比例（50%）

    # 评分：时间星级阈值（相对 time_limit 的倍率）
    # 例：<=1.00x -> 5星；<=1.10x -> 4星；<=1.25x -> 3星；<=1.50x -> 2星；>1.50x -> 1星
    time_star_cutoffs: tuple[float, float, float, float] = (1.00, 1.10, 1.25, 1.50)

    # 评分：食物温度阈值（绝对偏差，单位 °C）
    # 单件食物：|temp - serving| <= 4°C 计 5；<=7°C 计 4；<=10°C 计 3；<=15°C 计 2；更差 1
    temp_delta_cutoffs_c: tuple[float, float, float, float] = (4.0, 7.0, 10.0, 15.0)

    # 评分：从取餐到送达的时长阈值（秒）
    # 单件食物：<= 10min 计 5；<= 18min 计 4；<= 25min 计 3；<= 35min 计 2；更差 1
    freshness_cutoffs_s: tuple[float, float, float, float] = (600.0, 1080.0, 1500.0, 2100.0)

    # 食物综合打分：温度权重 & 新鲜度权重
    food_temp_weight: float = 0.5
    food_fresh_weight: float = 0.5

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
    """给定 ratio 与 4 段阈值，返回 5..1 星"""
    c1, c2, c3, c4 = cutoffs
    if ratio <= c1: return 5
    if ratio <= c2: return 4
    if ratio <= c3: return 3
    if ratio <= c4: return 2
    return 1

def _star_from_abs(x: float, cutoffs: tuple[float, float, float, float], reverse: bool = False) -> int:
    """
    给定绝对量与阈值，返回 5..1 星。
    reverse=False：小于等于 c1 为 5 星（越小越好），>c4 为 1 星
    """
    c1, c2, c3, c4 = cutoffs
    if not reverse:
        if x <= c1: return 5
        if x <= c2: return 4
        if x <= c3: return 3
        if x <= c4: return 2
        return 1
    # 反向未使用，这里保留接口
    if x >= c1: return 5
    if x >= c2: return 4
    if x >= c3: return 3
    if x >= c4: return 2
    return 1

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)

# ---------------- Public API ----------------

def compute_settlement(
    *,
    order_base_earnings: float,
    duration_s: float,
    time_limit_s: float,
    items: Iterable[Any],
    config: Optional[SettlementConfig] = None,
) -> SettlementResult:
    """
    计算一次订单送达的结算结果。

    参数：
    - order_base_earnings: 订单“写好”的基础工资
    - duration_s: 本单累计“活动用时”（虚拟时间，DeliveryMan 已在轮询里累计）
    - time_limit_s: 本单时限（没有就传 0）
    - items: 食物列表（需要尽量包含属性：temp_c, serving_temp_c, picked_at_sim, delivered_at_sim）
    - config: 可选配置（不传用默认）
    """
    cfg = config or SettlementConfig()

    # ---------- Base pay ----------
    base_pay = float(order_base_earnings)
    base_note = "on time"
    overtime_s = 0.0
    if time_limit_s and duration_s:
        if duration_s <= time_limit_s + cfg.overtime_grace_s:
            base_pay = order_base_earnings
            base_note = "on time"
        else:
            overtime_s = duration_s - time_limit_s
            if cfg.enable_base_overtime_penalty:
                # 线性折减：按超时比例往下打折，保底 min_base_fraction
                ratio = duration_s / max(1e-9, time_limit_s)
                fraction = max(cfg.overtime_linear_min_base_fraction, 1.0 / ratio)
                base_pay = order_base_earnings * fraction
                base_note = f"overtime ratio={ratio:.2f}, fraction={fraction:.2f}"
            else:
                # 仍给满基薪
                base_pay = order_base_earnings
                base_note = "overtime (no base penalty)"

    # ---------- Time star ----------
    if time_limit_s > 0:
        time_ratio = duration_s / max(1e-9, time_limit_s)
        time_star = _star_from_ratio(time_ratio, cfg.time_star_cutoffs)
    else:
        time_star = 5  # 无时限默认满分

    # ---------- Food star ----------
    # 对每件食物：由温度偏差 + 新鲜时长得到 1..5 星；取平均后再四舍五入
    per_item_stars = []
    for it in items or []:
        temp_c = _safe_float(getattr(it, "temp_c", None), default=math.nan)
        serving_c = _safe_float(getattr(it, "serving_temp_c", None), default=math.nan)
        picked_at = _safe_float(getattr(it, "picked_at_sim", None), default=math.nan)
        delivered_at = _safe_float(getattr(it, "delivered_at_sim", None), default=math.nan)

        # 温度：若缺数据，则忽略温度分；有就按偏差
        temp_star = None
        if not math.isnan(temp_c) and not math.isnan(serving_c):
            delta = abs(temp_c - serving_c)
            temp_star = _star_from_abs(delta, cfg.temp_delta_cutoffs_c)

        # 新鲜度：以“取餐到送达时长”为准；缺数据则忽略
        fresh_star = None
        if not math.isnan(picked_at) and not math.isnan(delivered_at):
            stay_s = max(0.0, delivered_at - picked_at)
            fresh_star = _star_from_abs(stay_s, cfg.freshness_cutoffs_s)

        # 融合
        if (temp_star is None) and (fresh_star is None):
            # 实在没数据，就给个中性 3 星
            per_item_stars.append(3)
        elif (temp_star is not None) and (fresh_star is not None):
            s = cfg.food_temp_weight * temp_star + cfg.food_fresh_weight * fresh_star
            per_item_stars.append(int(round(s)))
        else:
            per_item_stars.append(temp_star if temp_star is not None else fresh_star)

    if per_item_stars:
        food_star = int(round(sum(per_item_stars) / len(per_item_stars)))
        food_star = max(1, min(5, food_star))
    else:
        food_star = 3  # 没有条目就中性

    # ---------- Overall stars ----------
    # 简单：时间星与食物星取平均再四舍五入
    stars = int(round((time_star + food_star) / 2.0))
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
    elif stars == 1:
        extra = cfg.penalty_1_star
        extra_note = "1★ penalty"
    else:
        extra = cfg.penalty_0_star
        extra_note = "0★ penalty"

    total = float(base_pay) + float(extra)

    return SettlementResult(
        stars=stars,
        base_pay=float(base_pay),
        extra_pay=float(extra),
        total_pay=float(total),
        breakdown=dict(
            time=dict(duration_s=float(duration_s), time_limit_s=float(time_limit_s), overtime_s=float(overtime_s),
                      time_star=int(time_star), note=base_note),
            food=dict(per_item_stars=list(per_item_stars), food_star=int(food_star)),
            policy=dict(config=cfg.__dict__),
            extra_note=extra_note
        )
    )