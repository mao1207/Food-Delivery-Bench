# -*- coding: utf-8 -*-
# Communicator/unrealcv_delivery.py

import threading
from Communicator.unrealcv_basic import UnrealCV


class UnrealCvDelivery(UnrealCV):
    """
    中间层：提供 DeliveryMan 常用 BP 调用；所有 .client.request 已在基类串行化
    """

    def __init__(self, port, ip, resolution):
        super().__init__(port, ip, resolution)
        # 与基类同一把锁（RLock），避免多把锁嵌套
        self.lock = getattr(self, "_req_lock", None) or threading.RLock()

    # ---- BP 行为封装（带锁，语义清晰；基类已串行，所以这里即使省略 with 也安全） ----
    def d_move_forward(self, object_name):
        with self.lock:
            cmd = f"vbp {object_name} MoveForward"
            self.client.request(cmd)

    def d_rotate(self, object_name, angle, direction="left"):
        if direction == "right":
            clockwise = 1
        else:
            angle = -angle
            clockwise = -1
        with self.lock:
            cmd = f"vbp {object_name} Rotate_Angle {1} {angle} {clockwise}"
            self.client.request(cmd)

    def d_turn_around(self, object_name, angle, direction="left"):
        if direction == "right":
            clockwise = 1
        else:
            angle = -angle
            clockwise = -1
        with self.lock:
            cmd = f"vbp {object_name} TurnAround {angle} {clockwise}"
            self.client.request(cmd)

    def d_step_forward(self, object_name):
        with self.lock:
            cmd = f"vbp {object_name} StepForward"
            self.client.request(cmd)

    def d_stop(self, object_name):
        with self.lock:
            cmd = f"vbp {object_name} StopDeliveryMan"
            self.client.request(cmd)

    def d_get_on_a_bike(self, object_name):
        with self.lock:
            cmd = f"vbp {object_name} GetOnScooter"
            self.client.request(cmd)

    def s_set_state(self, object_name, throttle, brake, steering):
        with self.lock:
            cmd = f"vbp {object_name} SetState {throttle} {brake} {steering}"
            self.client.request(cmd)

    def get_informations(self, manager_object_name):
        with self.lock:
            cmd = f"vbp {manager_object_name} GetDeliveryInformation"
            return self.client.request(cmd)

    def update_agents(self, manager_object_name):
        with self.lock:
            cmd = f"vbp {manager_object_name} UpdateAgents"
            self.client.request(cmd)

    def making_u_turn(self, object_name):
        with self.lock:
            cmd = f"vbp {object_name} MakingUTurn"
            self.client.request(cmd)

    def d_pick_up(self, d_name, object_name):
        with self.lock:
            cmd = f"vbp {d_name} PickUp {object_name}"
            self.client.request(cmd)

    def get_camera_observation(self, camera_id, viewmode="lit"):
        # 直接走基类 read_image；已串行 & 返回 numpy(BGR)
        return self.read_image(camera_id, viewmode, "fast")
