# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from utils.map_debug_viewer import MapDebugViewer

# ------------------ 订单元数据转换 ------------------
class _Pos:
    def __init__(self, x: float, y: float):
        self.x = float(x); self.y = float(y)

class _NodeStub:
    def __init__(self, x: float, y: float):
        self.position = _Pos(x, y)
        self.type = "normal"

def _xy_from(obj: Any):
    if obj is None: return None
    try: return float(obj.position.x), float(obj.position.y)  # node-like
    except Exception: pass
    try: return float(obj.x), float(obj.y)                    # address-like
    except Exception: pass
    try: return float(obj["x"]), float(obj["y"])              # dict-like
    except Exception: return None

def _order_meta_from(orders: List[Any]) -> List[Dict[str, Any]]:
    metas: List[Dict[str, Any]] = []
    for o in (orders or []):
        try:
            oid = getattr(o, "id", None)
            pu_node = getattr(o, "pickup_node", None)
            do_node = getattr(o, "dropoff_node", None)

            if pu_node is None:
                pu_xy = _xy_from(getattr(o, "pickup_address", None))
                if pu_xy: pu_node = _NodeStub(*pu_xy)
            if do_node is None:
                do_xy = _xy_from(getattr(o, "delivery_address", None))
                if do_xy: do_node = _NodeStub(*do_xy)

            if pu_node is None and isinstance(o, dict) and "pickup_xy" in o:
                x, y = o["pickup_xy"]; pu_node = _NodeStub(x, y)
            if do_node is None and isinstance(o, dict) and "dropoff_xy" in o:
                x, y = o["dropoff_xy"]; do_node = _NodeStub(x, y)

            meta: Dict[str, Any] = {"id": oid}
            if pu_node is not None: meta["pickup_node"] = pu_node
            if do_node is not None: meta["dropoff_node"] = do_node
            metas.append(meta)
        except Exception:
            continue
    return metas


# ------------------ 导出器（单线程） ------------------
class MapExportor:
    """
    单线程离屏导出两张 PNG（global/local）。
    用法：
      1) exportor = MapExportor(map_obj=city_map, world_json_path=..., show_road_names=False)
      2) exportor.prepare_base()   # 只调用一次：构建静态底图
      3) 每次决策：g_bytes, l_bytes = exportor.export(agent_xy=(x, y), orders=orders)

    说明：
      - 与 UI 完全解耦：内部自建一个隐藏的 MapDebugViewer（真离屏），不会影响你窗口；
      - 底图只在 prepare_base() 时绘制；export() 仅更新动态层并导出；
      - 观感与 MapDebugViewer 保持一致（线宽/字号/抗锯齿/坐标轴/路名等）。
    """
    def __init__(self, *, map_obj: Any, world_json_path: Optional[str] = None,
                 show_road_names: bool = False, viewer: Optional[MapDebugViewer] = None):
        app = QApplication.instance()
        if app is None:
            raise RuntimeError("QApplication not initialized. Create it in main() first and run its event loop.")

        self.map = map_obj
        self.world_json_path = world_json_path
        self.show_road_names_default = bool(show_road_names)

        if viewer is None:
            self.viewer = MapDebugViewer(title="Exporter (headless)")
            # 真离屏：不在屏幕显示
            self.viewer.setAttribute(Qt.WA_DontShowOnScreen, True)
            self.viewer.hide()
        else:
            self.viewer = viewer

        self._base_ready = False

    # ---- 只需调用一次：构建静态底图 ----
    def prepare_base(self):
        self.viewer.prepare_export_base(map_obj=self.map, world_json_path=self.world_json_path)
        self._base_ready = True

    # ---- 每次决策：输出两张 PNG（bytes）----
    def export(self, *, agent_xy: Tuple[float, float], orders: List[Any]) -> Tuple[bytes, bytes]:
        if not self._base_ready:
            raise RuntimeError("MapExportor.export: export base not prepared. Call prepare_base() first.")

        ax, ay = float(agent_xy[0]), float(agent_xy[1])

        # 临时把订单元数据挂到 map（供“动态层”使用）；导出后恢复
        had_attr = hasattr(self.map, "order_meta")
        prev_val = getattr(self.map, "order_meta", None)
        setattr(self.map, "order_meta", _order_meta_from(orders))

        try:
            # 这里保持与原有算法一致（你已有的可达集函数）
            get_reach = getattr(self.map, "get_reachable_set_xy", None)
            reachable = get_reach(ax, ay) if callable(get_reach) else {"next_hop": [], "next_intersections": []}

            # 更新 agent 坐标并导出两张图（仅动态层操作；底图不重建）
            self.viewer.set_agent_xy(ax, ay)
            return self.viewer.generate_images(reachable, show_road_names=self.show_road_names_default)

        finally:
            # 恢复 map.order_meta，避免副作用
            if had_attr:
                setattr(self.map, "order_meta", prev_val)
            else:
                try: delattr(self.map, "order_meta")
                except Exception: pass