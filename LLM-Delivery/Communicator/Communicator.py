# -*- coding: utf-8 -*-
"""
Communicator.py
- 单实例（通常 9000 端口）供多个 agent 共享
- ✅ 所有 UE 调用统一加锁：基类已猴补；此处 _send_lock = self.lock（同一把 RLock）
- 每个 delivery_man_id 独立线程异步走路（go_to_xy_async），实现多 agent 并发
- get_location / get_orientation 的稳健解析（list/tuple/ndarray/字符串/"{}"）
- 支持 go_to_xy(..., speed_cm_s, accel_cm_s2, decel_cm_s2, arrive_tolerance_cm)
"""

import math
import random
import re
import threading
import time
from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple

from Communicator.unrealcv_delivery import UnrealCvDelivery
from Base.Types import Vector
from Config import Config

# 近似“瞬启/瞬停”的默认加速度与制动（cm/s^2）
DEFAULT_MAX_ACCEL_CM_S2   = 80000.0
DEFAULT_BRAKE_DECEL_CM_S2 = 80000.0

# 速度/加速度的“保活”写回周期（目前不再使用；保留常量以兼容旧代码）
SPEED_KEEPALIVE_SEC = 0.8


class Communicator(UnrealCvDelivery):
    def __init__(self, port: int, ip: str, resolution: Tuple[int, int]):
        # 统一参数顺序：与 unrealcv_basic 保持 (port, ip, resolution)
        super().__init__(port, ip, resolution)

        self.delivery_manager_name: Optional[str] = None
        self.delivery_man_id_to_name: Dict[str, str] = {}

        self._walkers: Dict[str, threading.Thread] = {}
        self._stops: Dict[str, threading.Event] = {}
        self._lock = threading.RLock()

        # ✅ 与中间层同一把锁（RLock）
        self._send_lock = self.lock

    # ------------------------------------------------------------------ 基础映射
    def get_delivery_man_name(self, agent_id: Any) -> str:
        key = str(agent_id)
        name = self.delivery_man_id_to_name.get(key)
        if not name:
            name = f"GEN_DELIVERY_MAN_{key}"
            self.delivery_man_id_to_name[key] = name
        return name

    # ------------------------------------------------------------------ 生成实体
    def spawn_delivery_man(self, agent_id: Any, x: float, y: float) -> None:
        key = str(agent_id)
        name = f"GEN_DELIVERY_MAN_{key}"
        self.delivery_man_id_to_name[key] = name

        model_name = Config.DELIVERY_MAN_MODEL_PATH
        with self._send_lock:
            self.spawn_bp_asset(model_name, name)
            self.set_location((float(x), float(y), 110.0), name)
            self.set_orientation((0.0, 0.0, 0.0), name)
            self.set_scale((1.0, 1.0, 1.0), name)
            self.set_collision(name, True)
            self.set_movable(name, True)

    def spawn_delivery_manager(self) -> None:
        self.delivery_manager_name = "GEN_DeliveryManager"
        with self._send_lock:
            self.spawn_bp_asset(Config.DELIVERY_MANAGER_MODEL_PATH, self.delivery_manager_name)

    def spawn_customer(self, order_id: int, x: float, y: float) -> None:
        name = f"GEN_CUSTOMER_{order_id}"
        with self._send_lock:
            self.spawn_bp_asset(Config.CUSTOMER_MODEL_PATH, name)
            self.set_location((float(x), float(y), 110.0), name)
            self.set_orientation((0.0, random.uniform(0, 360), 0.0), name)
            self.set_scale((1.0, 1.0, 1.0), name)
            self.set_collision(name, True)
            self.set_movable(name, True)

    def destroy_customer(self, order_id: int) -> None:
        name = f"GEN_CUSTOMER_{order_id}"
        with self._send_lock:
            self.destroy(name)

    # ----------------------------------------------------------- 解析工具 + 就绪等待
    def _parse_vec3(self, raw: Any) -> Optional[Tuple[float, float, float]]:
        if raw is None:
            return None

        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            try:
                vals = list(raw)
                if len(vals) >= 2:
                    x = float(vals[0]); y = float(vals[1])
                    z = float(vals[2]) if len(vals) > 2 else 0.0
                    return (x, y, z)
            except Exception:
                return None

        if hasattr(raw, "__iter__") and not isinstance(raw, (str, bytes, dict)):
            try:
                it = iter(raw)
                x = float(next(it)); y = float(next(it))
                try:
                    z = float(next(it))
                except StopIteration:
                    z = 0.0
                return (x, y, z)
            except Exception:
                pass

        if isinstance(raw, str):
            s = raw.strip()
            if not s or s == "{}":
                return None

            mX = re.search(r"[Xx]\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", s)
            mY = re.search(r"[Yy]\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", s)
            mZ = re.search(r"[Zz]\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", s)
            if mX and mY:
                x = float(mX.group(1)); y = float(mY.group(1))
                z = float(mZ.group(1)) if mZ else 0.0
                return (x, y, z)

            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
            if len(nums) >= 2:
                x = float(nums[0]); y = float(nums[1])
                z = float(nums[2]) if len(nums) > 2 else 0.0
                return (x, y, z)

        return None

    def wait_actor_ready(self, delivery_man_id: Any, timeout: float = 5.0, interval: float = 0.1) -> bool:
        name = self.get_delivery_man_name(delivery_man_id)
        end = time.time() + float(timeout)
        while time.time() < end:
            with self._send_lock:
                raw_loc = self.get_location(name)
                raw_rot = self.get_orientation(name)
            loc = self._parse_vec3(raw_loc)
            rot = self._parse_vec3(raw_rot)
            if loc and rot:
                return True
            time.sleep(interval)
        return False

    # ------------------------------------------------------------------ 位置 / 朝向
    def get_position_and_direction(self, delivery_man_id: Any) -> Dict[str, Tuple[Vector, float]]:
        try:
            name = self.get_delivery_man_name(delivery_man_id)
            with self._send_lock:
                raw_loc = self.get_location(name)
                raw_rot = self.get_orientation(name)

            loc = self._parse_vec3(raw_loc)
            rot = self._parse_vec3(raw_rot)
            if not loc or not rot:
                print(f"Warning: Could not retrieve location/orientation for {name}")
                return {}

            px, py = float(loc[0]), float(loc[1])
            yaw = float(rot[1]) if len(rot) >= 2 else 0.0  # pitch,yaw,roll -> 取 yaw

            position = Vector(px, py)
            return {str(delivery_man_id): (position, yaw)}

        except Exception as e:
            print(f"Error in get_position_and_direction: {e}")
            return {}

    # ------------------------------------------------------------------ 常用动作（统一加锁）
    def set_orientation_safe(self, name: str, euler_xyz: List[float]) -> None:
        with self._send_lock:
            self.set_orientation(euler_xyz, name)

    def delivery_man_move_forward(self, delivery_man_id: Any) -> None:
        name = self.get_delivery_man_name(delivery_man_id)
        with self._send_lock:
            self.d_move_forward(name)

    def delivery_man_stop(self, delivery_man_id: Any) -> None:
        name = self.get_delivery_man_name(delivery_man_id)
        with self._send_lock:
            self.d_stop(name)

    def delivery_man_step_forward(self, delivery_man_id: Any, speed: float, time: float) -> None:   
        name = self.get_delivery_man_name(delivery_man_id)
        with self._send_lock:
            self.d_step_forward(name, speed, time)

    def delivery_man_turn_around(self, delivery_man_id: Any, angle: float, direction: str) -> None:
        name = self.get_delivery_man_name(delivery_man_id)
        with self._send_lock:
            self.d_turn_around(name, angle, direction)

    # ------------------------------------------------------------ 速度/加速度配置
    def configure_speed_profile(
        self,
        delivery_man_id: Any,
        speed_cm_s: Optional[float] = None,
        accel_cm_s2: Optional[float] = None,
        decel_cm_s2: Optional[float] = None,
    ) -> None:
        """便捷设置速度、加速度、制动。未传则用较大的默认值以近似匀速。"""
        try:
            with self._send_lock:
                if speed_cm_s is not None:
                    self.set_max_speed(delivery_man_id, float(speed_cm_s))
                self.set_max_accel(
                    delivery_man_id,
                    float(accel_cm_s2 if accel_cm_s2 is not None else DEFAULT_MAX_ACCEL_CM_S2),
                )
                self.set_braking_decel(
                    delivery_man_id,
                    float(decel_cm_s2 if decel_cm_s2 is not None else DEFAULT_BRAKE_DECEL_CM_S2),
                )
        except Exception as e:
            print(f"[UE] configure_speed_profile failed on {delivery_man_id}: {e}")

    # ------------------------------------------------------------------ 路线行走（异步）
    def stop_go_to(self, delivery_man_id: Any) -> None:
        key = str(delivery_man_id)
        with self._lock:
            ev = self._stops.get(key)
            if ev:
                ev.set()

    def go_to_xy_async(
        self,
        delivery_man_id: Any,
        route: List[Tuple[float, float]],
        speed_cm_s: Optional[float] = None,
        accel_cm_s2: Optional[float] = None,
        decel_cm_s2: Optional[float] = None,
        arrive_tolerance_cm: float = 300.0,
    ) -> None:
        """非阻塞：为该 agent 启动一个后台线程并行走整条路线。"""
        if not route or len(route) < 1:
            return

        key = str(delivery_man_id)
        self.stop_go_to(key)

        stop_event = threading.Event()
        t = threading.Thread(
            target=self._walk_route,
            args=(key, list(route), stop_event, speed_cm_s, accel_cm_s2, decel_cm_s2, float(arrive_tolerance_cm)),
            name=f"UEWalker-{key}",
            daemon=True,
        )
        with self._lock:
            self._stops[key] = stop_event
            self._walkers[key] = t
        t.start()

    # 兼容旧接口
    def go_to_xy(
        self,
        delivery_man_id: Any,
        route: List[Tuple[float, float]],
        speed_cm_s: Optional[float] = None,
        accel_cm_s2: Optional[float] = None,
        decel_cm_s2: Optional[float] = None,
        arrive_tolerance_cm: float = 300.0,
    ) -> None:
        self.go_to_xy_async(
            delivery_man_id,
            route,
            speed_cm_s=speed_cm_s,
            accel_cm_s2=accel_cm_s2,
            decel_cm_s2=decel_cm_s2,
            arrive_tolerance_cm=arrive_tolerance_cm,
        )

    def teleport_xy(self, delivery_man_id: Any, x: float, y: float) -> None:
        name = self.get_delivery_man_name(delivery_man_id)

        # 停掉 Python 侧走路线程
        self.stop_go_to(delivery_man_id)

        # ✅ 等待线程退出
        t = self._walkers.pop(str(delivery_man_id), None)
        if t and t.is_alive():
            try:
                t.join(timeout=0.2)
            except Exception:
                pass

        # UE 停止移动
        try:
            with self._send_lock:
                self.d_stop(name)
        except Exception:
            pass

        # 执行瞬移
        cmd = f"vbp {name} SetLocation {float(x):.2f} {float(y):.2f}"
        with self._send_lock:
            self.client.request(cmd)



    def _walk_route(
        self,
        delivery_man_id: str,
        route: List[Tuple[float, float]],
        stop_event: threading.Event,
        speed_cm_s: Optional[float],
        accel_cm_s2: Optional[float],
        decel_cm_s2: Optional[float],
        arrive_tolerance_cm: float,
    ) -> None:
        """
        极简 Walk 版（依赖 UE 的 Orient Rotation to Movement 自动转向）：
        - 路线开始仅配置一次速度/加速度/制动
        - 每个路点只下发一次 MoveToNextPoint
        - 轮询距离，进容差即切下一点
        - 不做朝向纠偏、不做速度保活
        """
        name = self.get_delivery_man_name(delivery_man_id)
        tol = float(arrive_tolerance_cm)

        def _get_pos() -> Optional["Vector"]:
            cur = self.get_position_and_direction(delivery_man_id)
            if not cur:
                return None
            pos, _ = cur.get(delivery_man_id, (None, None))
            return pos

        def _dist_to(tx: float, ty: float) -> Optional[float]:
            p = _get_pos()
            if not p:
                return None
            return math.hypot(float(p.x) - float(tx), float(p.y) - float(ty))

        def _issue_move_xy_accept(x: float, y: float, accept: float) -> None:
            cmd = f"vbp {name} MoveToNextPoint {float(x):.2f} {float(y):.2f} {float(accept):.1f}"
            with self._send_lock:
                self.client.request(cmd)

        # 路线开始：仅配置一次移动参数
        self.configure_speed_profile(
            delivery_man_id,
            speed_cm_s=speed_cm_s,
            accel_cm_s2=accel_cm_s2,
            decel_cm_s2=decel_cm_s2,
        )

        try:
            for (tx, ty) in route:
                if stop_event.is_set():
                    break

                _issue_move_xy_accept(tx, ty, tol)

                # 轮询直到进入容差
                stagnant_count = 0
                max_stagnant = 10          # 连续几次无进展就退出
                eps = 1e-3                 # 认为“没变化”的距离阈值（可调）
                prev_d = None

                while not stop_event.is_set():
                    d = _dist_to(tx, ty)

                    # 1) 到达判定
                    if d is not None and d <= tol:
                        break

                    # 2) 无进展判定
                    if d is None:
                        stagnant_count = 0               # 无法计算距离就不记停滞
                    else:
                        if prev_d is not None and abs(d - prev_d) <= eps:
                            stagnant_count += 1
                        else:
                            stagnant_count = 0
                        prev_d = d

                        if stagnant_count >= max_stagnant:
                            print(f"[{name}] no progress for {max_stagnant} ticks (|Δd|≤{eps}); break to avoid stall.")
                            break

                    time.sleep(0.05)


            # # 收尾：保险停一下
            # self.delivery_man_stop(delivery_man_id)

        except Exception as e:
            try:
                self.delivery_man_stop(delivery_man_id)
            except Exception:
                pass
            print(f"[{name}] walk_route error: {e}")


    # def _walk_route(
    #     self,
    #     delivery_man_id: str,
    #     route: List[Tuple[float, float]],
    #     stop_event: threading.Event,
    #     speed_cm_s: Optional[float],
    #     accel_cm_s2: Optional[float],
    #     decel_cm_s2: Optional[float],
    #     arrive_tolerance_cm: float,
    # ) -> None:
    #     """
    #     DEBUG 版：忽略 route/速度/轮询，直接把角色移动到 (0,0,当前Z)。
    #     """
    #     # 角色名 & 容差
    #     name = self.get_delivery_man_name(delivery_man_id)
    #     tol = float(arrive_tolerance_cm)

    #     # 取当前 Z（失败就用固定兜底值，避免穿地）
    #     tz = 110.0
    #     try:
    #         with self._send_lock:
    #             raw_loc = self.get_location(name)
    #         v = self._parse_vec3(raw_loc)  # 期望返回 (x, y, z)
    #         if v:
    #             tz = float(v[2])
    #     except Exception:
    #         pass

    #     # 只发一次 MoveToPoint
    #     cmd = f'vbp {name} MoveToNextPoint {0.0:.2f} {0.0:.2f} {0.0:.2f}'
    #     with self._send_lock:
    #         resp = self.client.request(cmd)
    #     print(f"[DEBUG] issued: {cmd} -> {resp}")





