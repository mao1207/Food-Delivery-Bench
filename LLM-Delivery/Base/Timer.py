# timer.py
# 纯 Python 虚拟时间，不依赖 Qt。用于让 viewer/UE 的 tick 用同一条“虚拟时间轴”推进。
# - VirtualClock: 可设 time_scale；提供 now_sim() / sim_to_real_duration()。
# - TimeCursor: 面向每个使用者（viewer/UE/AI）的“光标”，便于取 dt_sim。

import time
from dataclasses import dataclass

class VirtualClock:
    """
    虚拟时间轴：
      now_sim = base_sim + (monotonic() - base_real) * time_scale
    修改 time_scale 会保持 now_sim 连续。
    """
    def __init__(self, time_scale: float = 1.0):
        self._time_scale = float(time_scale)
        self._base_real = time.monotonic()
        self._base_sim = 0.0  # 从 0 开始

    @property
    def time_scale(self) -> float:
        return self._time_scale

    def set_time_scale(self, scale: float):
        scale = float(scale)
        if scale <= 0:
            scale = 1e-9
        # 保持 now_sim 连续
        now_r = time.monotonic()
        now_s = self.now_sim()
        self._base_real = now_r
        self._base_sim = now_s
        self._time_scale = scale

    def now_sim(self) -> float:
        return self._base_sim + (time.monotonic() - self._base_real) * self._time_scale

    def sim_to_real_duration(self, sim_seconds: float) -> float:
        return float(sim_seconds) / max(1e-9, self._time_scale)

    def make_cursor(self) -> "TimeCursor":
        return TimeCursor(self)


@dataclass
class TimeCursor:
    clock: VirtualClock
    _last_sim: float = None

    def __post_init__(self):
        self._last_sim = self.clock.now_sim()

    def dt(self) -> float:
        """返回自上次调用以来的虚拟秒，并推进光标。"""
        now_s = self.clock.now_sim()
        dt_s = max(0.0, now_s - self._last_sim)
        self._last_sim = now_s
        return dt_s

    def now(self) -> float:
        return self.clock.now_sim()
