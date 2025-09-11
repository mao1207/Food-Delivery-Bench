# -*- coding: utf-8 -*-
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

class ScooterState(str, Enum):
    USABLE   = "usable"    # 可骑
    DEPLETED = "depleted"  # 没电，不可骑
    PARKED   = "parked"    # 停车/抛车（静止在某处）

@dataclass
class EScooter:
    """
    e-scooter 轻量模型：
    - 速度：cm/s
    - 电量：百分比 [0, 100]
    - 充电速率：%/min
    - 状态：USABLE / DEPLETED / PARKED
    - 停车坐标：park_xy（在 PARKED 状态下有效）
    """
    avg_speed_cm_s: float = 800.0
    min_speed_cm_s: float = 300.0
    max_speed_cm_s: float = 1500.0

    battery_max_pct: float = 100.0
    battery_pct: float = 100.0
    charge_rate_pct_per_min: float = 25.0

    state: ScooterState = ScooterState.USABLE
    park_xy: Optional[Tuple[float, float]] = None
    owner_id: str = ""
    with_owner: bool = True

    # === 速度相关 ===
    def clamp_speed(self, v: float) -> float:
        v = float(v)
        if v < self.min_speed_cm_s: return self.min_speed_cm_s
        if v > self.max_speed_cm_s: return self.max_speed_cm_s
        return v

    def set_speed(self, v: float) -> float:
        self.avg_speed_cm_s = self.clamp_speed(v)
        return self.avg_speed_cm_s

    # === 电量相关 ===
    def set_battery_pct(self, pct: float) -> float:
        p = float(pct)
        if p < 0.0: p = 0.0
        if p > self.battery_max_pct: p = self.battery_max_pct
        self.battery_pct = p
        # 同步状态
        if self.state != ScooterState.PARKED:
            self.state = ScooterState.DEPLETED if self.battery_pct <= 0.0 else ScooterState.USABLE
        return self.battery_pct

    def consume_pct(self, delta_pct: float) -> float:
        """扣除电量百分点，并据此更新状态。"""
        return self.set_battery_pct(self.battery_pct - float(delta_pct))

    def charge_to(self, target_pct: float) -> float:
        """充电到指定百分比（立即生效；上层可用 WAIT 模拟时间消耗）。"""
        return self.set_battery_pct(target_pct)

    # === 停车 / 取车 ===
    def park_here(self, x: float, y: float):
        self.state = ScooterState.PARKED
        self.park_xy = (float(x), float(y))

    def unpark(self):
        """取车后变为 USABLE/DEPLETED（取决于电量），清空停车坐标。"""
        self.park_xy = None
        self.state = ScooterState.DEPLETED if self.battery_pct <= 0.0 else ScooterState.USABLE

    def can_be_ridden_by(self, agent_id) -> bool:
        return str(agent_id) == str(self.owner_id) if self.owner_id else True
