# -*- coding: utf-8 -*-
# Communicator/unrealcv_basic.py

import time
import threading
import json
import numpy as np
import PIL.Image
from io import BytesIO

try:
    import cv2
except Exception:
    cv2 = None

import unrealcv


class UnrealCV(object):
    """
    UnrealCV 基类：
    - 单 TCP 连接，所有 request/request_batch 通过一把 RLock 串行化（防止 msg_id 乱序）
    - decode_png / decode_bmp 返回 numpy(BGR) 以便前端统一显示
    """

    def __init__(self, port, ip, resolution):
        self.ip = ip
        self.client = unrealcv.Client((ip, port))
        self.client.connect()

        # === 统一请求锁 & 猴补 ===
        self._req_lock = getattr(self, "_req_lock", None) or threading.RLock()
        self._orig_request = self.client.request
        self._orig_request_batch = getattr(self.client, "request_batch", None)

        def _locked_request(cmd, timeout=None):
            with self._req_lock:
                if timeout is None:
                    return self._orig_request(cmd)
                else:
                    return self._orig_request(cmd, timeout)

        def _locked_request_batch(cmds):
            with self._req_lock:
                return self._orig_request_batch(cmds)

        self.client.request = _locked_request
        if self._orig_request_batch:
            self.client.request_batch = _locked_request_batch

        self.resolution = resolution
        self.ini_unrealcv(resolution)

    # -------------------- UnrealCV 初始化 --------------------
    def ini_unrealcv(self, resolution=(320, 240)):
        self.check_connection()
        w, h = resolution
        self.client.request(f"vrun setres {w}x{h}w", -1)
        self.client.request("DisableAllScreenMessages", -1)
        self.client.request("vrun sg.ShadowQuality 0", -1)
        self.client.request("vrun sg.TextureQuality 0", -1)
        self.client.request("vrun sg.EffectsQuality 0", -1)
        time.sleep(0.1)
        self.client.message_handler = self.message_handler

    def message_handler(self, message):
        # 预留：需要时可解析服务端消息
        return message

    def check_connection(self):
        while self.client.isconnected() is False:
            print("UnrealCV server is not running. Please try again")
            time.sleep(1)
            self.client.connect()

    # -------------------- 物体/Actor 操作 --------------------
    def spawn(self, prefab, name):
        cmd = f"vset /objects/spawn {prefab} {name}"
        self.client.request(cmd)

    def spawn_bp_asset(self, prefab_path, name):
        cmd = f"vset /objects/spawn_bp_asset {prefab_path} {name}"
        self.client.request(cmd)

    def clean_garbage(self):
        self.client.request("vset /action/clean_garbage")

    def set_location(self, loc, name):
        x, y, z = loc
        cmd = f"vset /object/{name}/location {x} {y} {z}"
        self.client.request(cmd)

    def set_orientation(self, orientation, name):
        pitch, yaw, roll = orientation
        cmd = f"vset /object/{name}/rotation {pitch} {yaw} {roll}"
        self.client.request(cmd)

    def set_scale(self, scale, name):
        x, y, z = scale
        cmd = f"vset /object/{name}/scale {x} {y} {z}"
        self.client.request(cmd)

    def set_physics(self, actor_name, hasPhysics: bool):
        cmd = f"vset /object/{actor_name}/physics {hasPhysics}"
        self.client.request(cmd)

    def set_collision(self, actor_name, hasCollision: bool):
        cmd = f"vset /object/{actor_name}/collision {hasCollision}"
        self.client.request(cmd)

    def set_movable(self, actor_name, isMovable: bool):
        cmd = f"vset /object/{actor_name}/object_mobility {isMovable}"
        self.client.request(cmd)

    def destroy(self, actor_name):
        cmd = f"vset /object/{actor_name}/destroy"
        self.client.request(cmd)

    # 这些接口在上层 Communicator.configure_speed_profile 会调用
    def set_max_speed(self, delivery_man_id, speed_cm_s: float) -> bool:
        name = self.get_delivery_man_name(delivery_man_id)  # 由子类提供
        cmd = f"vbp {name} SetMaxSpeed {float(speed_cm_s)}"
        self.client.request(cmd)

    def set_max_accel(self, delivery_man_id, accel_cm_s2: float) -> bool:
        name = self.get_delivery_man_name(delivery_man_id)
        cmd = f"vbp {name} SetMaxAccel {float(accel_cm_s2)}"
        self.client.request(cmd)

    def set_braking_decel(self, delivery_man_id, decel_cm_s2: float) -> bool:
        name = self.get_delivery_man_name(delivery_man_id)
        cmd = f"vbp {name} SetBrakingDel {float(decel_cm_s2)}"
        self.client.request(cmd)

    # -------------------- 位置/姿态获取 --------------------
    def get_location(self, actor_name):
        try:
            cmd = f"vget /object/{actor_name}/location"
            res = self.client.request(cmd)
            location = [float(i) for i in res.split()]
            return np.array(location)
        except Exception as e:
            print(f"Error occurred in get_location: {e}")
            try:
                print("res:", res)
            except Exception:
                pass

    def get_location_batch(self, actor_names):
        cmd = [f"vget /object/{actor_name}/location" for actor_name in actor_names]
        res = self.client.request_batch(cmd)
        return [np.array([float(i) for i in r.split()]) for r in res]

    def get_orientation(self, actor_name):
        try:
            cmd = f"vget /object/{actor_name}/rotation"
            res = self.client.request(cmd)
            orientation = [float(i) for i in res.split()]
            return np.array(orientation)
        except Exception as e:
            print(f"Error occurred in get_orientation: {e}")
            try:
                print("res:", res)
            except Exception:
                pass

    def get_orientation_batch(self, actor_names):
        cmd = [f"vget /object/{actor_name}/rotation" for actor_name in actor_names]
        res = self.client.request_batch(cmd)
        return [np.array([float(i) for i in r.split()]) for r in res]

    # -------------------- 图像读取 --------------------
    def read_image(self, cam_id, viewmode, mode="direct"):
        # cam_id: 0,1,2... ; viewmode: lit/normal/depth/object_mask
        if mode == "direct":  # PNG
            cmd = f"vget /camera/{cam_id}/{viewmode} png"
            return self.decode_png(self.client.request(cmd))
        elif mode == "file":  # 保存到文件再读（不推荐）
            cmd = f"vget /camera/{cam_id}/{viewmode} {viewmode}{self.ip}.png"
            img_path = self.client.request(cmd)
            if cv2 is not None:
                return cv2.imread(img_path)  # BGR
            pil = PIL.Image.open(img_path).convert("RGB")
            arr = np.array(pil)[:, :, ::-1]  # to BGR
            return arr
        elif mode == "fast":  # BMP（快）
            cmd = f"vget /camera/{cam_id}/{viewmode} bmp"
            return self.decode_bmp(self.client.request(cmd))
        else:
            return None

    def decode_png(self, res):  # -> numpy BGR
        img = np.asarray(PIL.Image.open(BytesIO(res)))  # RGBA or RGB
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        # to BGR
        return img[:, :, ::-1].copy()

    def decode_bmp(self, res):  # -> numpy BGR
        pil = PIL.Image.open(BytesIO(res))
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        arr = np.array(pil)  # RGB
        return arr[:, :, ::-1].copy()  # BGR
