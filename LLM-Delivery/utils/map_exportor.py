# utils/map_exportor.py
# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QCoreApplication, Qt

from utils.map_debug_viewer import MapDebugViewer


def _ensure_qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        raise RuntimeError("QApplication not initialized. Create it in main() first.")
    return app


class _Pos:
    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)


class _NodeStub:
    def __init__(self, x: float, y: float):
        self.position = _Pos(x, y)
        self.type = "normal"


def _xy_from(obj: Any) -> Optional[Tuple[float, float]]:
    if obj is None:
        return None
    # node-like
    try:
        return float(obj.position.x), float(obj.position.y)
    except Exception:
        pass
    # address-like
    try:
        return float(obj.x), float(obj.y)
    except Exception:
        pass
    # dict-like
    try:
        return float(obj["x"]), float(obj["y"])
    except Exception:
        return None


def _order_meta_from(orders: List[Any]) -> List[Dict[str, Any]]:
    """
    允许多形态订单（对象/字典/仅有坐标）统一转成 viewer 可识别的结构。
    """
    metas: List[Dict[str, Any]] = []
    for o in (orders or []):
        try:
            oid = getattr(o, "id", None)
            pu_node = getattr(o, "pickup_node", None)
            do_node = getattr(o, "dropoff_node", None)

            # 兜底：从地址或 dict 构造 node stub
            if pu_node is None:
                pu_xy = _xy_from(getattr(o, "pickup_address", None))
                if pu_xy:
                    pu_node = _NodeStub(*pu_xy)
            if do_node is None:
                do_xy = _xy_from(getattr(o, "delivery_address", None))
                if do_xy:
                    do_node = _NodeStub(*do_xy)

            if pu_node is None and isinstance(o, dict) and "pickup_xy" in o:
                x, y = o["pickup_xy"]
                pu_node = _NodeStub(x, y)
            if do_node is None and isinstance(o, dict) and "dropoff_xy" in o:
                x, y = o["dropoff_xy"]
                do_node = _NodeStub(x, y)

            meta: Dict[str, Any] = {"id": oid}
            if pu_node is not None:
                meta["pickup_node"] = pu_node
            if do_node is not None:
                meta["dropoff_node"] = do_node
            metas.append(meta)
        except Exception:
            continue
    return metas


class MapExportor:
    """
    只做离屏导出两张 PNG（global/local），**只依赖 map.export_ns_xy(x, y)** 提供的
    N/S 数据。外部传入 viewer 时绝不改变其可见性/初始化状态；仅自建 viewer 时隐藏并
    draw_map 一次。
    """

    def __init__(
        self,
        *,
        map_obj: Any,
        viewer: Optional[MapDebugViewer] = None,
        world_json_path: Optional[str] = None,
        show_road_names: bool = False,
    ):
        _ensure_qapp()
        self.map = map_obj

        # 是否自建 viewer（仅自建时离屏隐藏并初始化场景）
        self._own_viewer: bool = viewer is None
        self.viewer = viewer or MapDebugViewer(title="Exporter(headless)")

        if self._own_viewer:
            self.viewer.setAttribute(Qt.WA_DontShowOnScreen, True)
            self.viewer.hide()

        self.world_json_path = world_json_path
        self.show_road_names = bool(show_road_names)
        self._inited_scene: bool = False  # 仅自建 viewer 有效

    def export(self, *, agent_xy: Tuple[float, float], orders: List[Any]) -> Tuple[bytes, bytes]:
        """
        渲染两张 PNG（bytes）：
          - Global：全局视图
          - Local ：以 agent 为中心的局部视图
        要求：self.map.export_ns_xy(x, y) -> {"next_hop":[...], "next_intersections":[...]}
              两个数组的 items 已带 label/label_text/x/y/dist_cm/type/road_name/roles 等。
        """
        # —— 临时挂订单到 map；渲染后恢复，避免副作用 ——
        had_attr = hasattr(self.map, "order_meta")
        prev_val = getattr(self.map, "order_meta", None)
        setattr(self.map, "order_meta", _order_meta_from(orders))

        try:
            # 自建 viewer 时只初始化一次场景
            if self._own_viewer and not self._inited_scene:
                self.viewer.draw_map(
                    self.map,
                    world_json_path=self.world_json_path,
                    show_bus=True,
                    show_docks=False,
                    show_building_links=False,
                    show_road_names=False,  # 具体路名显示交给 generate_images 参数
                    plain_mode="none",
                )
                self._inited_scene = True

            # 更新 agent 位置（你的 MapCanvasBase.set_agent_xy(x, y) 2 参版本）
            if agent_xy is not None and hasattr(self.viewer, "set_agent_xy"):
                ax, ay = agent_xy
                self.viewer.set_agent_xy(float(ax), float(ay))

            # === 仅调用 get_reachable_set_xy，作为 N/S 的唯一事实来源 ===

            # 离屏生成两张 PNG（bytes）
            g_bytes, l_bytes = self.viewer.generate_images(
                self.map.get_reachable_set_xy(float(agent_xy[0]), float(agent_xy[1])),
                show_road_names=self.show_road_names,
            )

            # 处理 Qt 事件队列（防止积压）
            QCoreApplication.processEvents()

            return g_bytes, l_bytes

        finally:
            # —— 还原 map.order_meta，保证对外无副作用 ——
            if had_attr:
                setattr(self.map, "order_meta", prev_val)
            else:
                try:
                    delattr(self.map, "order_meta")
                except AttributeError:
                    pass