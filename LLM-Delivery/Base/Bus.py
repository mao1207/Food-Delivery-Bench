# -*- coding: utf-8 -*-
# Base/Bus.py

import time
import math
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import List, Tuple, Optional, Dict, Any

from Base.Timer import VirtualClock

class BusState(str, Enum):
    STOPPED = "stopped"      # 在站点停靠
    MOVING = "moving"        # 在路线行驶
    WAITING = "waiting"      # 等待发车

@dataclass
class BusStop:
    """公交站点"""
    id: str
    x: float
    y: float
    name: str = ""
    wait_time_s: float = 10.0  # 停靠时间（秒）

@dataclass
class BusRoute:
    """公交路线"""
    id: str
    name: str
    stops: List[BusStop]
    path_points: List[Tuple[float, float]]  # 路线路径点
    speed_cm_s: float = 1200.0  # 行驶速度
    direction: int = 1  # 1: 正向, -1: 反向

@dataclass
class Bus:
    """公交车"""
    id: str
    route: BusRoute
    clock: VirtualClock = field(default_factory=lambda: VirtualClock())

    # 位置和状态
    x: float = 0.0
    y: float = 0.0
    state: BusState = BusState.WAITING

    # 路线进度
    current_stop_index: int = -1    # 当前站点（上一个到达的站点） current_stop_index + 1 是正在前往的站点
    current_path_index: int = 0     # 当前路径点（上一个到达的路径点）
    progress_to_next: float = 0.0  # 到下一个路径点的进度 [0,1]

    # 时间控制
    stop_start_time: float = 0.0
    last_update_time: float = 0.0

    # 乘客
    passengers: List[str] = field(default_factory=list)  # 乘客ID列表

    def __post_init__(self):
        # 初始化锁
        self._lock = Lock()

        # 初始化到第一个路径点（path[0]）
        if self.route.path_points:
            first_point = self.route.path_points[0]
            self.x = first_point[0]
            self.y = first_point[1]
            self.current_path_index = 0
            self.progress_to_next = 0.0
            self.state = BusState.MOVING  # 从第一个路径点开始移动
        elif self.route.stops:
            # 如果没有路径点，使用第一个站点
            first_stop = self.route.stops[0]
            self.x = first_stop.x
            self.y = first_stop.y
            self.state = BusState.STOPPED
            self.stop_start_time = self.clock.now_sim()

        self.last_update_time = self.clock.now_sim()

    def update(self):
        """更新公交状态和位置"""
        now = self.clock.now_sim()
        dt = now - self.last_update_time
        self.last_update_time = now

        if self.state == BusState.STOPPED:
            self._update_stopped(now)
        elif self.state == BusState.MOVING:
            self._update_moving(dt)

    def _update_stopped(self, now: float):
        """处理停靠状态"""
        if not self.route.stops:
            return

        current_stop = self.route.stops[self.current_stop_index]
        stop_duration = now - self.stop_start_time

        if stop_duration >= current_stop.wait_time_s:
            # 停靠时间到，准备出发
            self._depart_from_stop()

    def _update_moving(self, dt: float):
        """处理行驶状态"""
        if not self.route.path_points:
            return

        # 计算移动距离
        distance_cm = self.route.speed_cm_s * dt

        # 沿路径移动
        remaining_distance = distance_cm
        while remaining_distance > 0 and self.current_path_index < len(self.route.path_points) - 1:
            current_point = self.route.path_points[self.current_path_index]
            next_point = self.route.path_points[self.current_path_index + 1]

            # 计算到下一个点的距离
            dx = next_point[0] - current_point[0]
            dy = next_point[1] - current_point[1]
            segment_distance = math.hypot(dx, dy)

            if segment_distance == 0:  # 避免除零错误
                self.current_path_index += 1
                self.progress_to_next = 0.0
                continue

            if remaining_distance >= segment_distance * (1 - self.progress_to_next):
                # 够到达下一个点
                remaining_distance -= segment_distance * (1 - self.progress_to_next)
                self.current_path_index += 1
                self.progress_to_next = 0.0
            else:
                # 在当前段上前进
                self.progress_to_next += remaining_distance / segment_distance
                remaining_distance = 0.0

        if self.current_path_index >= len(self.route.path_points) - 1:
            # print(f"Bus {self.id} reached the end of the line and is returning.")

            # 反转站点和路径点列表
            self.route.stops.reverse()
            self.route.path_points.reverse()
            self.route.direction *= -1

            # 重置状态以从新的起点开始
            self.current_path_index = 0
            self.current_stop_index = -1
            self.progress_to_next = 0.0

        # 更新当前位置
        self._update_position()

        # 检查是否到达下一个站点
        self._check_arrival_at_stop()

    def _update_position(self):
        """根据路径进度更新位置"""
        if not self.route.path_points or self.current_path_index >= len(self.route.path_points):
            return

        if self.current_path_index < len(self.route.path_points) - 1:
            # 在两个路径点之间插值
            current_point = self.route.path_points[self.current_path_index]
            next_point = self.route.path_points[self.current_path_index + 1]

            self.x = current_point[0] + (next_point[0] - current_point[0]) * self.progress_to_next
            self.y = current_point[1] + (next_point[1] - current_point[1]) * self.progress_to_next
        else:
            # 在最后一个路径点
            last_point = self.route.path_points[-1]
            self.x = last_point[0]
            self.y = last_point[1]

    def _check_arrival_at_stop(self):
        """检查是否到达站点"""
        if not self.route.stops:
            return

        # 由于路径点是交错的（path[0], station0, path[1], station1, ...）
        # 站点在路径点中的索引是：1, 3, 5, ... (奇数位置)
        # 检查当前是否在站点对应的路径点位置
        if self.current_path_index % 2 == 1 and self.current_stop_index * 2 + 1 != self.current_path_index:  # 奇数位置是站点
            station_index = self.current_path_index // 2
            if station_index < len(self.route.stops):
                # 检查是否到达这个站点
                current_stop = self.route.stops[station_index]
                distance_to_stop = math.hypot(self.x - current_stop.x, self.y - current_stop.y)

                if distance_to_stop <= 300.0:  # 300cm = 3m 到达阈值
                    self._arrive_at_stop(station_index)

    def _arrive_at_stop(self, stop_index: int):
        """到达站点"""
        self.current_stop_index = stop_index
        self.state = BusState.STOPPED
        self.stop_start_time = self.clock.now_sim()

        # 精确停在站点位置
        stop = self.route.stops[stop_index]
        self.x = stop.x
        self.y = stop.y

        # 确保路径点索引指向正确的站点位置（奇数位置）
        expected_path_index = stop_index * 2 + 1
        if expected_path_index < len(self.route.path_points):
            self.current_path_index = expected_path_index

        # print(f"Bus {self.id} arrived at stop {stop.name or stop.id}")

    def _depart_from_stop(self):
        """从站点出发"""
        self.state = BusState.MOVING

        # 从当前站点出发，移动到下一个路径点
        # 当前在站点（奇数位置），下一个是路径点（偶数位置）
        if self.current_path_index < len(self.route.path_points) - 1:
            # self.current_path_index += 1
            self.progress_to_next = 0.0

        # print(f"Bus {self.id} departed from stop {self.route.stops[self.current_stop_index].name or self.route.stops[self.current_stop_index].id}")

    def get_next_stop(self) -> Optional[BusStop]:
        """获取下一个站点"""
        if not self.route.stops or self.current_stop_index + 1 >= len(self.route.stops):
            return None

        return self.route.stops[self.current_stop_index + 1]

    def get_current_stop(self) -> Optional[BusStop]:
        """获取当前站点"""
        if not self.route.stops or self.current_stop_index >= len(self.route.stops):
            return None
        return self.route.stops[self.current_stop_index]

    def is_at_stop(self) -> bool:
        """是否在站点停靠"""
        return self.state == BusState.STOPPED

    def board_passenger(self, passenger_id: str) -> bool:
        """乘客上车"""
        with self._lock:
            if self.is_at_stop() and passenger_id not in self.passengers:
                self.passengers.append(passenger_id)
                # print(f"Passenger {passenger_id} boarded bus {self.id}")
                return True
            return False

    def alight_passenger(self, passenger_id: str) -> bool:
        """乘客下车"""
        with self._lock:
            if passenger_id in self.passengers:
                self.passengers.remove(passenger_id)
                # print(f"Passenger {passenger_id} alighted from bus {self.id}")
                return True
            return False

    def _calculate_time_to_next_stop(self) -> Optional[float]:
        """计算到达下一站的时间（秒）"""
        if not self.route.stops or self.state != BusState.MOVING:
            return None

        next_stop = self.get_next_stop()
        if not next_stop:
            return None

        # 计算到下一站的距离
        distance_cm = math.hypot(next_stop.x - self.x, next_stop.y - self.y)

        if distance_cm > 0 and self.route.speed_cm_s > 0:
            return distance_cm / self.route.speed_cm_s

        return None

    def _get_remaining_stop_time(self) -> Optional[float]:
        """获取在当前站点的剩余停靠时间（秒）"""
        if self.state != BusState.STOPPED:
            return None

        current_stop = self.get_current_stop()
        if not current_stop:
            return None

        now = self.clock.now_sim()
        elapsed_time = now - self.stop_start_time
        remaining_time = current_stop.wait_time_s - elapsed_time

        # 返回剩余时间，如果已经超时则返回0
        return max(0.0, remaining_time)

    def get_status_text(self) -> str:
        """获取状态文本"""
        status_parts = [f"Bus {self.id}"]
        
        # 添加路线名称
        status_parts.append(f"route: {self.route.name}")

        if self.state == BusState.STOPPED:
            stop = self.get_current_stop()
            stop_name = stop.name if stop else "None"
            status_parts.append(f"stopped at {stop_name}")

            # 添加剩余停靠时间
            remaining_time = self._get_remaining_stop_time()
            if remaining_time is not None:
                status_parts.append(f"departing in: {remaining_time:.1f}s")

        elif self.state == BusState.MOVING:
            next_stop = self.get_next_stop()
            next_name = next_stop.name if next_stop else "None"
            status_parts.append(f"moving to {next_name}")

            # 添加到达时间
            time_to_next = self._calculate_time_to_next_stop()
            if time_to_next is not None:
                status_parts.append(f"ETA: {time_to_next:.1f}s")

        else:
            status_parts.append(f"state: {self.state.value}")

        # status_parts.append(f"passengers: {len(self.passengers)}")
        status_parts.append(f"pos: ({self.x/100:.1f}m, {self.y/100:.1f}m)")

        return " | ".join(status_parts)