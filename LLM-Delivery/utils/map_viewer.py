# # utils/map_viewer.py
# from typing import Dict, Any, List, Optional, Tuple, Callable
# import json, math, os, time, tempfile

# from PyQt5.QtWidgets import (
#     QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
#     QLabel, QTextEdit, QGraphicsRectItem, QPushButton, QLineEdit
# )
# from PyQt5.QtCore import Qt, QTimer, QCoreApplication
# from PyQt5.QtGui import QFont, QColor
# import pyqtgraph as pg
# from pyqtgraph.exporters import ImageExporter

# # =========================================================
# # =============== 1) 全局样式与参数（整洁版） ===============
# # =========================================================

# # ---------- 颜色 ----------
# COLOR_BG         = "#F7FBFF"
# COLOR_EDGE       = "#9EA7B3"
# COLOR_NODE       = "#5B9BD5"
# COLOR_HOP_WP     = "#1F7AFF"
# COLOR_NEXT_INT   = "#0066CC"
# COLOR_AGENT      = "#000000"

# COLOR_PICKUP     = "#FD9488"
# COLOR_DROPOFF    = "#2ECC71"

# COLOR_CHG        = "#27AE60"
# COLOR_BUS        = "#F39C12"
# COLOR_BUS_ROUTE  = "#FFA500"

# COLOR_AUX_EXT    = "#C7CDD4"   # 端点→延长垂足
# COLOR_AUX_PERP   = "#A2ACB3"   # 垂足→door/poi

# COLOR_BUILDING_PLAIN  = "#B0BEC5"  # 普通 building 填充灰
# PLAIN_FILL_ALPHA      = 80         # 普通 building 填充透明度
# COLOR_PLAIN_BORDER    = "#000000"  # 普通 building 黑色边框

# # 调试
# COLOR_DEBUG_NODE = "#00B8D9"
# COLOR_DEBUG_EDGE = "#C2185B"

# # ---------- 窗口（UI）尺寸/线宽 ----------
# UI_NODE_SIZE            = 10
# UI_NODE_SMALL_SIZE      = 10
# UI_HOP_SIZE             = 12
# UI_NINT_SIZE            = 14
# UI_POI_SIZE             = 9
# UI_AGENT_SIZE           = 26
# UI_AGENT_GLOBAL_SIZE    = 10
# UI_STAR_SIZE            = 36
# UI_LABEL_PX             = 16
# UI_BUILDING_ABBR_PX     = 13
# UI_EDGE_WIDTH           = 2.2
# UI_BUS_WIDTH            = 2.0
# UI_STAR_BORDER_W        = 1.6
# DEBUG_EDGE_WIDTH        = 4.0

# # ---------- 导出（EXPORT）尺寸/线宽 ----------
# EXP_EDGE_WIDTH          = 8.0
# EXP_BUS_WIDTH           = 8.0

# EXP_NODE_SIZE           = 5
# EXP_NODE_SMALL_SIZE     = 5
# EXP_HOP_SIZE            = 12
# EXP_NINT_SIZE           = 10
# EXP_POI_SIZE            = 6
# EXP_AGENT_SIZE_LOCAL    = 22
# EXP_AGENT_SIZE_GLOBAL   = 12

# # 导出：PU/DO、N/S 标签字号（全图/局部分开）
# EXP_LABEL_PX_GLOBAL     = 10
# EXP_LABEL_PX_LOCAL      = 13

# # 导出：路名字号（全图/局部分开）
# EXP_ROADNAME_PX_GLOBAL  = 9
# EXP_ROADNAME_PX_LOCAL   = 12

# # 导出：星标
# EXP_STAR_SIZE_GLOBAL    = 18
# EXP_STAR_SIZE_LOCAL     = 36
# EXP_STAR_BORDER_W       = 1.6

# # ---------- 导出清晰度与画幅 ----------
# TARGET_PX_PER_M_GLOBAL  = 3.2
# TARGET_PX_PER_M_LOCAL   = 6.0
# MIN_EXPORT_WIDTH_PX     = 1800
# MAX_EXPORT_WIDTH_PX     = 5200

# # ---------- 坐标轴 ----------
# AXIS_UNIT        = "m"
# CM_PER_UNIT      = 100.0
# TICK_STEP_UNIT   = 100.0
# AXIS_FONT_PX     = 14
# AXIS_LINE_WIDTH  = 2
# AXIS_MARGINS_PX  = (8, 8, 4, 18)

# # ---------- 标签（N/S/PU/DO）避让 ----------
# LABEL_BASE_N     = 200.0
# LABEL_STEP_N     = 70.0
# LABEL_BASE_PUDO  = 2600.0
# LABEL_STEP_PUDO  = 300.0
# LABEL_MAX_TRIES  = 24
# R_TEXT_BLACK     = 1400.0
# R_TEXT_COLOR     = 1200.0
# R_POINT_AVOID    = 900.0

# # ---------- 路名绘制 ----------
# ROAD_NAME_OFFSET_CM  = 140.0         # 相对道路中点的法向偏移
# ROAD_NAME_GRID_CM    = 1200.0        # 网格去重（导出）
# ROAD_NAME_PAD_SCALE  = 10.0          # 包围盒放大（更强避让）
# ROAD_NAME_TRIES      = 24
# ROAD_NAME_TSHIFT_CM  = 500.0         # 沿切向滑动

# # ---------- 视域 ----------
# LOCAL_MARGIN_CM  = 3500.0
# GLOBAL_PAD_CM    = 2500.0

# # ---------- 字体（字距压缩解决“003”松散） ----------
# UI_FONT_FAMILY              = "Arial"
# UI_LETTER_SPACING_PCT       = 92     # 100 正常；<100 更紧
# EXP_FONT_FAMILY             = "Arial"
# EXP_LETTER_SPACING_PCT      = 92

# # ---------- 建筑渲染 ----------
# BUILDING_TYPES = {"restaurant","store","rest_area","hospital","car_rental"}
# COLOR_BUILDING = {
#     "restaurant": "#E74C3C","store": "#3498DB","rest_area": "#9B59B6",
#     "hospital": "#E91E63","car_rental": "#1ABC9C",
# }
# ABBR = {"restaurant":"R","store":"S","rest_area":"A","hospital":"H","car_rental":"C"}

# # =========================================================
# # ====================== 2) 工具函数 ======================
# # =========================================================

# def _node_xy(node) -> Tuple[float, float]:
#     return float(node.position.x), float(node.position.y)

# def _is_road_node(nd) -> bool:
#     t = getattr(nd, "type", "")
#     return t in ("normal", "intersection")

# # =========================================================
# # ===================== 3) 主类 MapViewer =================
# # =========================================================

# class MapViewer(QMainWindow):
#     """绘底图、PU/DO、N/S、导出、动画 + 可选建筑连线显示 + 节点边调试"""
#     def __init__(self, title="Map Viewer"):
#         super().__init__()
#         self.setWindowTitle(title)

#         root = QWidget(); self.setCentralWidget(root)
#         v = QVBoxLayout(root)

#         # 顶栏
#         top = QHBoxLayout()
#         self.title_label = QLabel(title)
#         self.title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
#         top.addWidget(self.title_label)
#         top.addStretch(1)

#         self.btn_reinit = QPushButton("Reinit")
#         self.btn_save   = QPushButton("Save PNGs")
#         self.btn_start  = QPushButton("Start")
#         self.btn_pause  = QPushButton("Pause"); self.btn_pause.setEnabled(False)
#         for b in (self.btn_reinit, self.btn_save, self.btn_start, self.btn_pause):
#             top.addWidget(b)

#         # 输入目标坐标（动画/跳转）
#         self.input_x = QLineEdit(); self.input_x.setPlaceholderText("Target X (cm)")
#         self.input_y = QLineEdit(); self.input_y.setPlaceholderText("Target Y (cm)")
#         self.input_x.setFixedWidth(120); self.input_y.setFixedWidth(120)
#         self.btn_go = QPushButton("Go to (x, y)")
#         for w in (self.input_x, self.input_y, self.btn_go):
#             top.addWidget(w)

#         # 按坐标调试
#         self.inspect_x = QLineEdit(); self.inspect_x.setPlaceholderText("Inspect X (cm)")
#         self.inspect_y = QLineEdit(); self.inspect_y.setPlaceholderText("Inspect Y (cm)")
#         self.inspect_tol = QLineEdit(); self.inspect_tol.setPlaceholderText("Tol (cm)")
#         self.inspect_x.setFixedWidth(120); self.inspect_y.setFixedWidth(120); self.inspect_tol.setFixedWidth(90)
#         self.btn_inspect = QPushButton("Inspect")
#         for w in (self.inspect_x, self.inspect_y, self.inspect_tol, self.btn_inspect):
#             top.addWidget(w)

#         v.addLayout(top)

#         # 主画布
#         self.plot = pg.PlotWidget()
#         self.plot.setBackground(COLOR_BG)
#         self.plot.setAspectLocked(True)
#         self.plot.showGrid(x=True, y=True, alpha=0.15)
#         v.addWidget(self.plot)

#         # 控制台
#         self.info = QTextEdit(); self.info.setReadOnly(True); self.info.setFixedHeight(150)
#         v.addWidget(self.info)

#         # 状态
#         self._world_nodes: List[dict] = []
#         self._bus_paths: List[List[Tuple[float,float]]] = []
#         self._map = None
#         self._world_path = None
#         self._last_reachable: Optional[Dict[str,List[Dict[str,Any]]]] = None

#         self._label_font = QFont(); self._label_font.setBold(True); self._label_font.setPixelSize(UI_LABEL_PX)
#         self._bld_font   = QFont(self._label_font); self._bld_font.setPixelSize(UI_BUILDING_ABBR_PX)

#         self._agent_item = None
#         self._agent_xy: Optional[Tuple[float,float]] = None
#         self._taken: List[Tuple[float,float,float]] = []

#         # 路名（占位，可外部覆盖格式；最终显示为 <num><L/R>）
#         self._road_name_fmt: str = "{name}"

#         # 动画
#         self._anim_pts: List[Tuple[float,float]] = []
#         self._anim_speed = 10000.0
#         self._timer = QTimer(self); self._timer.setInterval(16); self._timer.timeout.connect(self._on_tick)
#         self._i = 0; self._done = 0.0

#         # 路径高亮
#         self._path_item = None
#         self._path_nodes_item = None

#         # 调试叠加图元
#         self._inspect_items: List[Any] = []

#         # 回调
#         self._on_reinit: Optional[Callable[[],None]] = None
#         self._on_go: Optional[Callable[[float, float], None]] = None
#         self._anim_done_cb: Optional[Callable[[float, float], None]] = None

#         # 绑定
#         self.btn_reinit.clicked.connect(self._handle_reinit)
#         self.btn_save.clicked.connect(self._handle_save)
#         self.btn_start.clicked.connect(self.start_animation)
#         self.btn_pause.clicked.connect(self.pause_animation)
#         self.btn_go.clicked.connect(self._handle_go_clicked)
#         self.btn_inspect.clicked.connect(self._handle_inspect_clicked)

#     # ---------------- 基础 ----------------
#     def reset_scene(self):
#         self.plot.clear()
#         self._agent_item = None
#         self._taken.clear()
#         self._last_reachable = None
#         self._path_item = None
#         self._path_nodes_item = None
#         self._clear_inspect_overlay()

#     def set_context(self, map_obj, world_json_path: Optional[str]):
#         self._map = map_obj
#         self._world_path = world_json_path

#     def set_reinit_callback(self, fn: Callable[[],None]): self._on_reinit = fn
#     def set_go_callback(self, fn: Callable[[float, float], None]): self._on_go = fn
#     def set_anim_done_callback(self, fn: Callable[[float, float], None]): self._anim_done_cb = fn
#     def get_agent_xy(self) -> Tuple[Optional[float], Optional[float]]:
#         return self._agent_xy if self._agent_xy else (None, None)

#     def _make_font(self, px: int, *, bold: bool = True, export: bool = False) -> QFont:
#         f = QFont()
#         f.setFamily(EXP_FONT_FAMILY if export else UI_FONT_FAMILY)
#         f.setPixelSize(int(px))
#         f.setBold(bold)
#         f.setKerning(True)
#         try:
#             f.setLetterSpacing(QFont.PercentageSpacing,
#                                EXP_LETTER_SPACING_PCT if export else UI_LETTER_SPACING_PCT)
#         except Exception:
#             pass
#         return f

#     # ---------------- 绘制 ----------------
#     def _draw_roads_only(self, target_plot, map_obj, edge_width: Optional[float]=None):
#         """仅画 kind∈{road,crosswalk,endcap}；屏蔽 aux_*"""
#         if edge_width is None:
#             edge_width = UI_EDGE_WIDTH
#         drawn = set()
#         for a, nbs in map_obj.adjacency_list.items():
#             for b in nbs:
#                 key = tuple(sorted([(id(a),), (id(b),)]))
#                 if key in drawn: continue
#                 drawn.add(key)
#                 get_meta = getattr(map_obj, "_get_edge_meta", None)
#                 meta = get_meta(a, b) if callable(get_meta) else {}
#                 kind = (meta.get("kind") or "") if isinstance(meta, dict) else ""
#                 if (not kind) or kind.startswith("aux_"): continue
#                 if kind not in ("road", "crosswalk", "endcap"): continue
#                 if not _is_road_node(a) or not _is_road_node(b): continue
#                 ax, ay = _node_xy(a); bx, by = _node_xy(b)
#                 target_plot.addItem(pg.PlotDataItem(
#                     x=[ax, bx], y=[ay, by],
#                     pen=pg.mkPen(COLOR_EDGE, width=edge_width),
#                     antialias=True
#                 ))

#     def draw_map(self, map_obj, world_json_path: Optional[str]=None,
#                  show_bus=True, show_docks=False, show_building_links: bool=False,
#                  show_road_names: bool=False, road_name_fmt: Optional[str]=None):
#         """窗口绘制（控制台）"""
#         self.set_context(map_obj, world_json_path)
#         if road_name_fmt: self._road_name_fmt = str(road_name_fmt)
#         self.reset_scene()

#         if world_json_path:
#             self._load_world(world_json_path)
#             if show_bus: self._draw_bus_routes(self.plot, bus_width=UI_BUS_WIDTH)
#             self._draw_buildings(self.plot)
#             if show_building_links:
#                 self._draw_all_building_boxes(self.plot)

#         self._draw_roads_only(self.plot, map_obj, edge_width=UI_EDGE_WIDTH)

#         # 点/节点
#         dock_set = getattr(map_obj, "_dock_nodes", set())
#         normals, docks, inters = [], [], []
#         for n in map_obj.nodes:
#             t = getattr(n, "type", "")
#             if t == "normal": (docks if n in dock_set else normals).append(_node_xy(n))
#             elif t == "intersection": inters.append(_node_xy(n))
#         if normals:
#             self.plot.addItem(pg.ScatterPlotItem(
#                 pos=normals, size=UI_NODE_SMALL_SIZE,
#                 brush=pg.mkBrush(COLOR_NODE), pen=pg.mkPen("w", width=0.3),
#                 symbol="o", antialias=True))
#         if show_docks and docks:
#             self.plot.addItem(pg.ScatterPlotItem(
#                 pos=docks, size=UI_NODE_SMALL_SIZE,
#                 brush=pg.mkBrush("#4C6EF5"), pen=pg.mkPen("w", width=0.3),
#                 symbol="o", antialias=True))
#         if inters:
#             self.plot.addItem(pg.ScatterPlotItem(
#                 pos=inters, size=UI_NODE_SIZE,
#                 brush=pg.mkBrush(COLOR_NODE), pen=pg.mkPen("w", width=0.5),
#                 symbol="o", antialias=True))

#         # 点状 POI
#         if getattr(map_obj, "pois", None):
#             chg, bus = [], []
#             for p in map_obj.pois:
#                 t = (p.type or "").lower()
#                 if t == "charging_station": chg.append(_node_xy(p))
#                 elif t == "bus_station":   bus.append(_node_xy(p))
#             if chg:
#                 self.plot.addItem(pg.ScatterPlotItem(
#                     pos=chg, size=UI_POI_SIZE,
#                     pen=pg.mkPen("w", width=1), brush=pg.mkBrush(COLOR_CHG),
#                     symbol="o", antialias=True))
#             if bus:
#                 self.plot.addItem(pg.ScatterPlotItem(
#                     pos=bus, size=UI_POI_SIZE+1,
#                     pen=pg.mkPen("#8E5A0A", width=1), brush=pg.mkBrush(COLOR_BUS),
#                     symbol="t", antialias=True))

#         # building links（aux_*）灰线仅在 True 时绘制
#         if show_building_links:
#             for (p1, p2, kind) in map_obj.get_building_link_segments():
#                 xs = [p1[0], p2[0]]; ys = [p1[1], p2[1]]
#                 color = COLOR_AUX_EXT if kind == "aux_ext" else COLOR_AUX_PERP
#                 self.plot.addItem(pg.PlotDataItem(
#                     x=xs, y=ys, pen=pg.mkPen(color, width=1.4, style=Qt.SolidLine),
#                     antialias=True))

#         # 路名（窗口）
#         if show_road_names:
#             self._draw_road_names(self.plot, px_per_cm=None, export=False, size_px=12)

#         self.info.setText("Map drawn.")

#         # Agent 黑点
#         if self._agent_xy is not None:
#             ax, ay = self._agent_xy
#             self._agent_item = pg.ScatterPlotItem(
#                 pos=[(ax, ay)], size=UI_AGENT_SIZE,
#                 brush=pg.mkBrush(COLOR_AGENT), pen=pg.mkPen("w", width=1.4),
#                 symbol="o", antialias=True)
#             self.plot.addItem(self._agent_item)

#     def mark_orders(self, map_obj):
#         for rec in getattr(map_obj, "order_meta", []):
#             oid = str(rec.get("id", ""))

#             # 灰框：订单里的 building
#             for key in ("pickup_building", "dropoff_building"):
#                 b = rec.get(key)
#                 if b and all(k in b for k in ("x", "y", "w", "h", "yaw")):
#                     x = float(b["x"]); y = float(b["y"])
#                     w = float(b["w"]); h = float(b["h"]); yaw = float(b["yaw"])
#                     rect = QGraphicsRectItem(x - w/2.0, y - h/2.0, w, h)
#                     rect.setTransformOriginPoint(x, y)
#                     rect.setRotation(yaw)
#                     rect.setZValue(-4)
#                     rect.setPen(pg.mkPen(COLOR_PLAIN_BORDER, width=1.2))
#                     fill_q = QColor(COLOR_BUILDING_PLAIN); fill_q.setAlpha(PLAIN_FILL_ALPHA)
#                     rect.setBrush(pg.mkBrush(fill_q))
#                     self.plot.addItem(rect)

#             # PU/DO 星号（窗口）
#             self._star_with_text(rec.get("pickup_node"),  f"P{oid}", COLOR_PICKUP,
#                                  size_px=UI_STAR_SIZE, border_w=UI_STAR_BORDER_W)
#             self._star_with_text(rec.get("dropoff_node"), f"D{oid}", COLOR_DROPOFF,
#                                  size_px=UI_STAR_SIZE, border_w=UI_STAR_BORDER_W)

#     def highlight_frontier(self, reachable: Dict[str,List[Dict[str,Any]]]):
#         self._last_reachable = reachable
#         for it in reachable.get("next_hop", []):
#             self._taken.append((float(it["x"]), float(it["y"]), R_POINT_AVOID))
#         for it in reachable.get("next_intersections", []):
#             self._taken.append((float(it["x"]), float(it["y"]), R_POINT_AVOID))
#         if self._agent_xy:
#             self._taken.append((self._agent_xy[0], self._agent_xy[1], R_POINT_AVOID))

#         # N：仅 waypoint
#         hops_all = reachable.get("next_hop", [])
#         hops_wp  = [it for it in hops_all if it.get("kind")=="waypoint" and not it.get("is_dock", False)]
#         if hops_wp:
#             self.plot.addItem(pg.ScatterPlotItem(pos=[(it["x"],it["y"]) for it in hops_wp], size=UI_HOP_SIZE,
#                                                  brush=pg.mkBrush(COLOR_HOP_WP), pen=pg.mkPen("w", width=0.8),
#                                                  symbol="o", antialias=True))
#         for it in hops_all:
#             lbl = it.get("label")
#             if not lbl: continue
#             lx, ly = self._place_label_live((float(it["x"]), float(it["y"])), True, LABEL_BASE_N, LABEL_STEP_N)
#             self._add_text(lbl, lx, ly, "#1F2D3D")

#         # S
#         ins = reachable.get("next_intersections", [])
#         if ins:
#             self.plot.addItem(pg.ScatterPlotItem(pos=[(it["x"],it["y"]) for it in ins], size=UI_NINT_SIZE,
#                                                  brush=pg.mkBrush(COLOR_NEXT_INT), pen=pg.mkPen("w", width=1.2),
#                                                  symbol="d", antialias=True))
#             for it in ins:
#                 lbl = it.get("label")
#                 if not lbl: continue
#                 lx, ly = self._place_label_live((float(it["x"]), float(it["y"])), True, LABEL_BASE_N, LABEL_STEP_N)
#                 self._add_text(lbl, lx, ly, "#1F2D3D")

#     def print_frontier_info(self, text: str):
#         self.info.setText(text if isinstance(text, str) and text.strip() else "(no text)")

#     # ---------------- Agent & 动画 ----------------
#     def set_agent_xy(self, x: float, y: float):
#         self._agent_xy = (x, y)
#         if self._agent_item is None:
#             self._agent_item = pg.ScatterPlotItem(pos=[(x,y)], size=UI_AGENT_SIZE,
#                                                   brush=pg.mkBrush(COLOR_AGENT), pen=pg.mkPen("w", width=1.4),
#                                                   symbol="o", antialias=True)
#             self.plot.addItem(self._agent_item)
#         else:
#             self._agent_item.setData(pos=[(x,y)])

#     def prepare_animation(self, pts: List[Tuple[float,float]], speed_cm_s: float = 10000.0):
#         self._anim_pts = list(pts or []); self._anim_speed = float(speed_cm_s)
#         self._i = 0; self._done = 0.0
#         if self._anim_pts:
#             x0, y0 = self._anim_pts[0]; self.set_agent_xy(x0, y0)
#         self.btn_start.setEnabled(True); self.btn_pause.setEnabled(False)

#     def start_animation(self):
#         if len(self._anim_pts) < 2: return
#         self._timer.start(); self.btn_start.setEnabled(False); self.btn_pause.setEnabled(True)

#     def pause_animation(self):
#         self._timer.stop(); self.btn_start.setEnabled(True); self.btn_pause.setEnabled(False)

#     def _on_tick(self):
#         if self._i >= len(self._anim_pts)-1:
#             self.pause_animation()
#             if self._anim_pts:
#                 fx, fy = self._anim_pts[-1]
#                 self.set_agent_xy(fx, fy)
#                 if self._anim_done_cb: self._anim_done_cb(fx, fy)
#             return
#         p0 = self._anim_pts[self._i]; p1 = self._anim_pts[self._i+1]
#         dx, dy = p1[0]-p0[0], p1[1]-p0[1]; L = math.hypot(dx,dy)
#         if L < 1e-6:
#             self._i += 1; self._done = 0.0; return
#         dt = self._timer.interval()/1000.0; self._done += self._anim_speed*dt
#         if self._done >= L:
#             self.set_agent_xy(p1[0],p1[1]); self._i += 1; self._done = 0.0
#         else:
#             t = self._done/L; self.set_agent_xy(p0[0]+dx*t, p0[1]+dy*t)

#     # ---------------- 路径高亮 ----------------
#     def clear_path_highlight(self):
#         if self._path_item is not None:
#             try: self.plot.removeItem(self._path_item)
#             except Exception: pass
#             self._path_item = None
#         if self._path_nodes_item is not None:
#             try: self.plot.removeItem(self._path_nodes_item)
#             except Exception: pass
#             self._path_nodes_item = None

#     def highlight_path(self, pts: List[Tuple[float, float]]):
#         if not pts or len(pts) < 2: return
#         xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
#         self._path_item = pg.PlotDataItem(x=xs, y=ys,
#                                           pen=pg.mkPen("#FF4757", width=3),
#                                           antialias=True)
#         self._path_item.setZValue(8)
#         self.plot.addItem(self._path_item)
#         self._path_nodes_item = pg.ScatterPlotItem(pos=list(zip(xs, ys)), size=5,
#                                                    brush=pg.mkBrush("#FF4757"),
#                                                    pen=pg.mkPen("w", width=0.6),
#                                                    symbol="o", antialias=True)
#         self._path_nodes_item.setZValue(9)
#         self.plot.addItem(self._path_nodes_item)

#     # =========================================================
#     # ============== 4) 导出：生成/保存分离接口 ===============
#     # =========================================================

#     def generate_images(self, reachable: Dict[str,List[Dict[str,Any]]],
#                         show_road_names: bool=False,
#                         road_name_fmt: Optional[str]=None) -> Tuple[bytes, bytes]:
#         """
#         返回 (global_png_bytes, local_png_bytes)，**不落盘**。
#         Agent 直接调用这个。
#         """
#         if self._map is None:
#             return (b"", b"")
#         if road_name_fmt: self._road_name_fmt = str(road_name_fmt)

#         # ---------- 全图 ----------
#         pw_g = self._new_canvas_export(aspect_locked=True)
#         self._draw_base(pw_g, draw_bus=True, draw_buildings=True, export=True)
#         self._draw_order_building_boxes(pw_g)

#         gxmin, gxmax, gymin, gymax = self._global_bounds()
#         span_x_g = max(1.0, (gxmax - gxmin))
#         px_per_cm_g_target = TARGET_PX_PER_M_GLOBAL / 100.0
#         width_px_g = int(span_x_g * px_per_cm_g_target)
#         width_px_g = max(MIN_EXPORT_WIDTH_PX, min(MAX_EXPORT_WIDTH_PX, width_px_g))
#         pw_g.plotItem.vb.setRange(xRange=(gxmin, gxmax), yRange=(gymin, gymax), padding=0)
#         QCoreApplication.processEvents()

#         # taken_rects：用于避让（先占 agent 圆）
#         taken_global: List[Tuple[float,float,float,float]] = []
#         def push_circle_rect(cx, cy, size_px, scale=0.6):
#             r_cm = (size_px * scale) / max(1.0, (width_px_g/span_x_g))
#             taken_global.append((cx-r_cm, cx+r_cm, cy-r_cm, cy+r_cm))

#         if self._agent_xy:
#             ax, ay = self._agent_xy
#             pw_g.addItem(pg.ScatterPlotItem(pos=[(ax, ay)], size=EXP_AGENT_SIZE_GLOBAL,
#                                             brush=pg.mkBrush(COLOR_AGENT), pen=pg.mkPen("w", width=1.0),
#                                             symbol="o", antialias=True))
#             push_circle_rect(ax, ay, EXP_AGENT_SIZE_GLOBAL)

#         # 先放 PU/DO（会把文本与圆加入 taken）
#         self._draw_orders_export(pw_g, px_per_cm=width_px_g/span_x_g, taken_rects=taken_global,
#                                  star_size_px=EXP_STAR_SIZE_GLOBAL, star_border_w=EXP_STAR_BORDER_W,
#                                  label_px=EXP_LABEL_PX_GLOBAL)

#         # 再放路名（避让上面已放置内容）
#         if show_road_names:
#             self._draw_road_names(pw_g, px_per_cm=width_px_g/span_x_g, export=True,
#                                   size_px=EXP_ROADNAME_PX_GLOBAL, taken_rects=taken_global)

#         self._apply_sparse_axes(pw_g, (gxmin, gxmax, gymin, gymax))
#         g_bytes = self._export(pw_g, None, (gxmin, gxmax, gymin, gymax),
#                                target_long_side_px=width_px_g, return_bytes=True)

#         # ---------- 局部 ----------
#         pw_l = self._new_canvas_export(aspect_locked=True)
#         self._draw_base(pw_l, draw_bus=False, draw_buildings=True, export=True)
#         self._draw_order_building_boxes(pw_l)

#         lxmin, lxmax, lymin, lymax = self._local_bounds(reachable)
#         span_x_l = max(1.0, (lxmax - lxmin))
#         px_per_cm_l_target = TARGET_PX_PER_M_LOCAL / 100.0
#         width_px_l = int(span_x_l * px_per_cm_l_target)
#         width_px_l = max(MIN_EXPORT_WIDTH_PX, min(MAX_EXPORT_WIDTH_PX, width_px_l))
#         pw_l.plotItem.vb.setRange(xRange=(lxmin, lxmax), yRange=(lymin, lymax), padding=0)
#         QCoreApplication.processEvents()

#         # 初始 taken：N/S/agent 的圆（用于避让）
#         taken_local: List[Tuple[float,float,float,float]] = []
#         def push_circle_rect_l(cx, cy, size_px, scale=0.6):
#             r_cm = (size_px * scale) / max(1.0, (width_px_l/span_x_l))
#             taken_local.append((cx-r_cm, cx+r_cm, cy-r_cm, cy+r_cm))

#         for it in reachable.get("next_hop", []):
#             ax, ay = float(it["x"]), float(it["y"]); push_circle_rect_l(ax, ay, 700.0)
#         for it in reachable.get("next_intersections", []):
#             ax, ay = float(it["x"]), float(it["y"]); push_circle_rect_l(ax, ay, 700.0)
#         if self._agent_xy:
#             ax, ay = self._agent_xy; push_circle_rect_l(ax, ay, 900.0)
#             pw_l.addItem(pg.ScatterPlotItem(pos=[(ax, ay)], size=EXP_AGENT_SIZE_LOCAL,
#                                             brush=pg.mkBrush(COLOR_AGENT),
#                                             pen=pg.mkPen("w", width=1.4),
#                                             symbol="o", antialias=True))

#         # 先放 N/S（会把文本加入 taken）
#         self._draw_ns(pw_l, reachable, with_labels=True, draw_poi_symbol=False,
#                       for_export=True, px_per_cm=width_px_l/span_x_l, taken_rects=taken_local,
#                       label_px=EXP_LABEL_PX_LOCAL)

#         # 再放 PU/DO（也加入 taken）
#         self._draw_orders_export(pw_l, px_per_cm=width_px_l/span_x_l, taken_rects=taken_local,
#                                  star_size_px=EXP_STAR_SIZE_LOCAL, star_border_w=EXP_STAR_BORDER_W,
#                                  label_px=EXP_LABEL_PX_LOCAL)

#         # 最后放路名（避让所有文字/圆）
#         if show_road_names:
#             self._draw_road_names(pw_l, px_per_cm=width_px_l/span_x_l, export=True,
#                                   size_px=EXP_ROADNAME_PX_LOCAL, taken_rects=taken_local)

#         self._apply_sparse_axes(pw_l, (lxmin, lxmax, lymin, lymax))
#         l_bytes = self._export(pw_l, None, (lxmin, lxmax, lymin, lymax),
#                                target_long_side_px=width_px_l, return_bytes=True)

#         return g_bytes, l_bytes

#     def save_images(self, reachable: Dict[str,List[Dict[str,Any]]],
#                     global_path: Optional[str]="global_map.png",
#                     local_path: Optional[str]="local_zoom.png",
#                     show_road_names: bool=False,
#                     road_name_fmt: Optional[str]=None) -> None:
#         """仅保存，不返回字节。内部复用 generate_images。"""
#         g_bytes, l_bytes = self.generate_images(reachable, show_road_names, road_name_fmt)
#         if global_path:
#             with open(global_path, "wb") as f: f.write(g_bytes or b"")
#             self.info.append(f"Saved: {os.path.abspath(global_path)}")
#         else:
#             self.info.append("Saved: (global skipped)")
#         if local_path:
#             with open(local_path, "wb") as f: f.write(l_bytes or b"")
#             self.info.append(f"Saved: {os.path.abspath(local_path)}")
#         else:
#             self.info.append("Saved: (local skipped)")

#     # 兼容旧名（可选）：render_pngs -> generate_images
#     def render_pngs(self, reachable: Dict[str,List[Dict[str,Any]]],
#                     show_road_names: bool=False, road_name_fmt: Optional[str]=None) -> Tuple[bytes, bytes]:
#         return self.generate_images(reachable, show_road_names, road_name_fmt)

#     # ---------------- 内部：绘制/布局工具 ----------------
#     def _new_canvas_export(self, aspect_locked: bool=True) -> pg.PlotWidget:
#         w = pg.PlotWidget()
#         w.setBackground(COLOR_BG)
#         w.setAspectLocked(aspect_locked)  # 关键：锁定 1:1
#         w.showGrid(x=True,y=True,alpha=0.15)
#         for side in ("left","right","top","bottom"):
#             w.getPlotItem().hideAxis(side)
#         try:
#             w.plotItem.layout.setContentsMargins(0,0,0,0)
#             w.plotItem.vb.setDefaultPadding(0.0)
#         except Exception:
#             pass
#         return w

#     def _apply_sparse_axes(self, pw: pg.PlotWidget, bounds: Tuple[float,float,float,float]):
#         xmin, xmax, ymin, ymax = bounds
#         pi = pw.getPlotItem()
#         for side in ("left", "right", "top", "bottom"):
#             pi.showAxis(side)
#             ax = pi.getAxis(side)
#             try:
#                 ax.setPen(pg.mkPen("k", width=AXIS_LINE_WIDTH))
#                 f = QFont(); f.setPixelSize(AXIS_FONT_PX)
#                 ax.setStyle(tickTextOffset=6, tickFont=f)
#                 ax.setTextPen(pg.mkPen("k"))
#             except Exception: pass
#         try:
#             pi.layout.setContentsMargins(*AXIS_MARGINS_PX)
#         except Exception: pass

#         step_cm = TICK_STEP_UNIT * CM_PER_UNIT
#         def gen_ticks(a_cm: float, b_cm: float) -> List[Tuple[float, str]]:
#             if step_cm <= 0: return []
#             k0 = math.ceil(a_cm / step_cm)
#             k1 = math.floor(b_cm / step_cm)
#             ticks = []
#             for k in range(int(k0), int(k1) + 1):
#                 pos_cm = k * step_cm
#                 val_unit = pos_cm / CM_PER_UNIT
#                 lab = f"{val_unit:.0f}" if abs(val_unit - round(val_unit)) < 1e-6 else f"{val_unit:.1f}"
#                 ticks.append((pos_cm, lab))
#             return ticks
#         ticks_x = gen_ticks(xmin, xmax); ticks_y = gen_ticks(ymin, ymax)
#         try:
#             pi.getAxis("bottom").setTicks([ticks_x])
#             pi.getAxis("top").setTicks([ticks_x])
#             pi.getAxis("left").setTicks([ticks_y])
#             pi.getAxis("right").setTicks([ticks_y])
#             pi.setLabel("bottom", f"x ({AXIS_UNIT})")
#             pi.setLabel("left",   f"y ({AXIS_UNIT})")
#         except Exception: pass

#     def _draw_base(self, pw: pg.PlotWidget, draw_bus: bool, draw_buildings: bool, export: bool=False):
#         if self._world_path:
#             self._load_world(self._world_path)
#             if draw_bus: self._draw_bus_routes(pw, bus_width=(EXP_BUS_WIDTH if export else UI_BUS_WIDTH))
#             if draw_buildings: self._draw_buildings(pw)

#         # 只画道路类边（过滤 aux_*）
#         self._draw_roads_only(pw, self._map, edge_width=(EXP_EDGE_WIDTH if export else UI_EDGE_WIDTH))

#         node_small = EXP_NODE_SMALL_SIZE if export else UI_NODE_SMALL_SIZE
#         node_main  = EXP_NODE_SIZE if export else UI_NODE_SIZE

#         normals, inters = [], []
#         for n in self._map.nodes:
#             t = getattr(n, "type", "")
#             if t == "normal": normals.append(_node_xy(n))
#             elif t == "intersection": inters.append(_node_xy(n))
#         if normals:
#             pw.addItem(pg.ScatterPlotItem(pos=normals, size=node_small,
#                                           brush=pg.mkBrush(COLOR_NODE), pen=pg.mkPen("w", width=0.3),
#                                           symbol="o", antialias=True))
#         if inters:
#             pw.addItem(pg.ScatterPlotItem(pos=inters, size=node_main,
#                                           brush=pg.mkBrush(COLOR_NODE), pen=pg.mkPen("w", width=0.5),
#                                           symbol="o", antialias=True))

#         if getattr(self._map,"pois",None):
#             chg, bus = [], []
#             for p in self._map.pois:
#                 t = (p.type or "").lower()
#                 if t == "charging_station": chg.append(_node_xy(p))
#                 elif t == "bus_station":   bus.append(_node_xy(p))
#             poi_sz = EXP_POI_SIZE if export else UI_POI_SIZE
#             if chg:
#                 pw.addItem(pg.ScatterPlotItem(pos=chg, size=poi_sz,
#                                               pen=pg.mkPen("w", width=1), brush=pg.mkBrush(COLOR_CHG),
#                                               symbol="o", antialias=True))
#             if bus:
#                 pw.addItem(pg.ScatterPlotItem(pos=bus, size=poi_sz+1,
#                                               pen=pg.mkPen("#8E5A0A", width=1), brush=pg.mkBrush(COLOR_BUS),
#                                               symbol="t", antialias=True))

#     def _draw_order_building_boxes(self, pw: pg.PlotWidget):
#         """导出时把订单里的灰色建筑也画出来"""
#         if not getattr(self._map, "order_meta", None): return
#         for rec in self._map.order_meta:
#             for key in ("pickup_building", "dropoff_building"):
#                 b = rec.get(key)
#                 if not b or not all(k in b for k in ("x","y","w","h","yaw")): continue
#                 x = float(b["x"]); y = float(b["y"])
#                 w = float(b["w"]); h = float(b["h"]); yaw = float(b["yaw"])
#                 rect = QGraphicsRectItem(x - w/2.0, y - h/2.0, w, h)
#                 rect.setTransformOriginPoint(x, y); rect.setRotation(yaw); rect.setZValue(-4)
#                 rect.setPen(pg.mkPen(COLOR_PLAIN_BORDER, width=1.2))
#                 fill_q = QColor(COLOR_BUILDING_PLAIN); fill_q.setAlpha(PLAIN_FILL_ALPHA)
#                 rect.setBrush(pg.mkBrush(fill_q)); pw.addItem(rect)

#     # ---- 路名：拆 side，做 <num><L/R> ----
#     def _split_name_side(self, meta: dict) -> Tuple[str, Optional[str]]:
#         import re
#         raw_name = str(meta.get("name") or "").strip()
#         side_raw = str(meta.get("side") or "").strip().lower()
#         side = side_raw if side_raw in ("left", "right") else None
#         if side is None:
#             m = re.search(r"\((left|right)\)", raw_name, flags=re.I)
#             if m: side = m.group(1).lower()
#         base = re.sub(r"\([^)]*\)", "", raw_name).strip()
#         base = re.sub(r"\b(road|rd|street|st|avenue|ave|boulevard|blvd|drive|dr|lane|ln|way|place|pl|court|ct|terrace|ter)\b\.?",
#                       "", base, flags=re.I).strip()
#         return base, side

#     def _lr_label(self, base: str, side: Optional[str]) -> str:
#         import re
#         m = re.search(r"(\d+)", base or "")
#         num = m.group(1) if m else (re.sub(r"[^A-Za-z]", "", base)[:4] or "?").upper()
#         suf = "L" if side == "left" else ("R" if side == "right" else "?")
#         return f"{num}{suf}"

#     def _draw_road_names(self, pw: pg.PlotWidget, px_per_cm: Optional[float],
#                          export: bool, size_px: Optional[int] = None,
#                          taken_rects: Optional[List[Tuple[float,float,float,float]]] = None):
#         """
#         每个 (基础路名, 侧别) 只放一个标签，位置=长度加权全局中点；文本=<数字><L/R>。
#         导出模式下：网格去重 + 包围盒避让（可与外部 taken_rects 联动）。
#         """
#         if self._map is None: return
#         get_meta = getattr(self._map, "_get_edge_meta", None)

#         # 聚合
#         agg: Dict[Tuple[str,str], Dict[str, float]] = {}
#         seen_pairs = set()
#         for a, nbs in self._map.adjacency_list.items():
#             for b in nbs:
#                 keyp = tuple(sorted((id(a), id(b))))
#                 if keyp in seen_pairs: continue
#                 seen_pairs.add(keyp)

#                 if not _is_road_node(a) or not _is_road_node(b): continue
#                 meta = get_meta(a, b) if callable(get_meta) else {}
#                 if not isinstance(meta, dict) or (meta.get("kind") != "road"): continue

#                 base, side = self._split_name_side(meta)
#                 if not base or side not in ("left", "right"):
#                     continue

#                 ax, ay = _node_xy(a); bx, by = _node_xy(b)
#                 dx, dy = bx - ax, by - ay
#                 L = math.hypot(dx, dy)
#                 if L < 1e-6: continue
#                 mx, my = (ax + bx) / 2.0, (ay + by) / 2.0

#                 rec = agg.setdefault((base, side), dict(sumL=0.0, cx=0.0, cy=0.0, vx=0.0, vy=0.0))
#                 rec["sumL"] += L
#                 rec["cx"]   += mx * L
#                 rec["cy"]   += my * L
#                 rec["vx"]   += dx
#                 rec["vy"]   += dy

#         if not agg: return

#         # 文本盒
#         def _text_box_cm(text: str) -> Tuple[float, float]:
#             px = size_px if size_px is not None else (EXP_ROADNAME_PX_GLOBAL if export else 12)
#             w_px = max(1.0, len(text) * px * 0.6)
#             h_px = px * 1.2
#             if not px_per_cm: return 1.0, 1.0
#             return (w_px / px_per_cm) * ROAD_NAME_PAD_SCALE, (h_px / px_per_cm) * ROAD_NAME_PAD_SCALE

#         def _collide(rect, pool):
#             if not pool: return False
#             x0, x1, y0, y1 = rect
#             for a0, a1, b0, b1 in pool:
#                 if not (x1 <= a0 or a1 <= x0 or y1 <= b0 or b1 <= y0):
#                     return True
#             return False

#         pool = taken_rects if (export and px_per_cm and taken_rects is not None) else []

#         grid_seen = set()

#         for (base, side), rec in agg.items():
#             sumL = rec["sumL"]
#             if sumL <= 0: continue
#             mx = rec["cx"] / sumL
#             my = rec["cy"] / sumL
#             vx, vy = rec["vx"], rec["vy"]
#             vlen = math.hypot(vx, vy)
#             if vlen < 1e-6: tx, ty = 1.0, 0.0
#             else: tx, ty = vx / vlen, vy / vlen
#             nx, ny = -ty, tx

#             label = self._lr_label(base, side)

#             if export and px_per_cm:
#                 gx = int(round(mx / ROAD_NAME_GRID_CM))
#                 gy = int(round(my / ROAD_NAME_GRID_CM))
#                 tag = (label, gx, gy)
#                 if tag in grid_seen: continue
#                 grid_seen.add(tag)

#             sgn = -1 if side == "left" else 1
#             offsets = [ROAD_NAME_OFFSET_CM * k for k in (1.0, 1.4, 1.8, 2.2)]
#             slides  = [0.0, ROAD_NAME_TSHIFT_CM, -ROAD_NAME_TSHIFT_CM,
#                        2*ROAD_NAME_TSHIFT_CM, -2*ROAD_NAME_TSHIFT_CM]

#             w_cm, h_cm = _text_box_cm(label)
#             placed = False; tries = 0
#             for off in offsets:
#                 for sl in slides:
#                     px_ = mx + sgn * nx * off + tx * sl
#                     py_ = my + sgn * ny * off + ty * sl
#                     rect = (px_ - w_cm/2, px_ + w_cm/2, py_ - h_cm/2, py_ + h_cm/2)
#                     if not _collide(rect, pool):
#                         if pool is not None: pool.append(rect)
#                         placed = True; break
#                     tries += 1
#                     if tries >= ROAD_NAME_TRIES: break
#                 if placed or (tries >= ROAD_NAME_TRIES): break
#             if not placed:
#                 px_ = mx + sgn * nx * offsets[0]
#                 py_ = my + sgn * ny * offsets[0]

#             ti = pg.TextItem(text=label, color="#000000", anchor=(0.5,0.5))
#             px_use = size_px if size_px is not None else (EXP_ROADNAME_PX_GLOBAL if export else 12)
#             ti.setFont(self._make_font(px_use, bold=True, export=export))
#             ti.setPos(px_, py_); ti.setZValue(11)
#             pw.addItem(ti)

#     def _text_rect_cm(self, text: str, px_per_cm: float, px: int) -> Tuple[float,float]:
#         w_px = max(1.0, len(text) * px * 0.6)
#         h_px = px * 1.2
#         PAD = 4.0
#         return (w_px / px_per_cm) * PAD, (h_px / px_per_cm) * PAD

#     def _overlap_rect(self, a, b) -> bool:
#         ax0, ax1, ay0, ay1 = a; bx0, bx1, by0, by1 = b
#         return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)

#     def _draw_ns(self, pw: pg.PlotWidget, reachable, with_labels: bool,
#                  draw_poi_symbol: bool, for_export: bool=False,
#                  px_per_cm: float=1.0,
#                  taken_rects: Optional[List[Tuple[float,float,float,float]]]=None,
#                  label_px: int = EXP_LABEL_PX_GLOBAL):
#         hops_all = reachable.get("next_hop", [])
#         hops_wp  = [it for it in hops_all if it.get("kind")=="waypoint" and not it.get("is_dock", False)]

#         size_hop = EXP_HOP_SIZE if for_export else UI_HOP_SIZE
#         size_nint= EXP_NINT_SIZE if for_export else UI_NINT_SIZE

#         if hops_wp:
#             pw.addItem(pg.ScatterPlotItem(pos=[(it["x"],it["y"]) for it in hops_wp], size=size_hop,
#                                           brush=pg.mkBrush(COLOR_HOP_WP), pen=pg.mkPen("w", width=0.8),
#                                           symbol="o", antialias=True))
#         if draw_poi_symbol:
#             hops_poi = [it for it in hops_all if it.get("kind")=="poi"]
#             if hops_poi:
#                 pw.addItem(pg.ScatterPlotItem(pos=[(it["x"],it["y"]) for it in hops_poi], size=size_hop,
#                                               brush=pg.mkBrush("#20BF6B"), pen=pg.mkPen("w", width=0.8),
#                                               symbol="o", antialias=True))
#         ins = reachable.get("next_intersections", [])
#         if ins:
#             pw.addItem(pg.ScatterPlotItem(pos=[(it["x"],it["y"]) for it in ins], size=size_nint,
#                                           brush=pg.mkBrush(COLOR_NEXT_INT), pen=pg.mkPen("w", width=1.2),
#                                           symbol="d", antialias=True))

#         if not with_labels: return

#         if not for_export:
#             for it in hops_all + ins:
#                 lbl = it.get("label")
#                 if not lbl: continue
#                 lx, ly = self._place_label_live((float(it["x"]), float(it["y"])), True, LABEL_BASE_N, LABEL_STEP_N)
#                 self._add_text(lbl, lx, ly, "#1F2D3D", target_plot=pw)
#             return

#         # 导出：严格 bbox 避让
#         if taken_rects is None: taken_rects = []
#         def place_bb(ax: float, ay: float, text: str, base: float, step: float) -> Tuple[float,float,Tuple[float,float,float,float]]:
#             w_cm, h_cm = self._text_rect_cm(text, px_per_cm, label_px)
#             dirs = [(1,1),(-1,1),(1,-1),(-1,-1),(0,1),(0,-1),(1,0),(-1,0)]
#             for k in range(LABEL_MAX_TRIES):
#                 d = base + k*step
#                 for dx,dy in dirs:
#                     cx = ax + dx*d; cy = ay + dy*d
#                     rect = (cx - w_cm/2, cx + w_cm/2, cy - h_cm/2, cy + h_cm/2)
#                     if all(not self._overlap_rect(rect, r) for r in taken_rects):
#                         taken_rects.append(rect); return cx, cy, rect
#             cx = ax + base; cy = ay + base
#             rect = (cx - w_cm/2, cx + w_cm/2, cy - h_cm/2, cy + h_cm/2)
#             taken_rects.append(rect); return cx, cy, rect

#         def add_text(lbl,x,y):
#             ti = pg.TextItem(text=lbl, color="#1F2D3D", anchor=(0.5,0.5))
#             ti.setFont(self._make_font(label_px, bold=True, export=True))
#             ti.setPos(x,y); ti.setZValue(12); pw.addItem(ti)

#         for it in hops_all + ins:
#             lbl = it.get("label")
#             if not lbl: continue
#             cx, cy, _ = place_bb(float(it["x"]), float(it["y"]), lbl, LABEL_BASE_N, LABEL_STEP_N)
#             add_text(lbl, cx, cy)

#     # === 导出：PU/DO（矩形避让） ===
#     def _draw_orders_export(self, pw: pg.PlotWidget, px_per_cm: float,
#                             taken_rects: List[Tuple[float,float,float,float]],
#                             star_size_px: int, star_border_w: float = EXP_STAR_BORDER_W,
#                             label_px: int = EXP_LABEL_PX_GLOBAL):
#         if not getattr(self._map, "order_meta", None): return

#         def push_circle_rect(cx, cy, size_px, scale=0.6):
#             r_cm = (size_px * scale) / max(1.0, px_per_cm)
#             taken_rects.append((cx-r_cm, cx+r_cm, cy-r_cm, cy+r_cm))

#         def text_rect(text: str):
#             w_cm, h_cm = self._text_rect_cm(text, px_per_cm, label_px)
#             return w_cm, h_cm

#         def place_bb(ax: float, ay: float, text: str, base: float, step: float):
#             w_cm, h_cm = text_rect(text)
#             dirs = [(1,1),(-1,1),(1,-1),(-1,-1),(0,1),(0,-1),(1,0),(-1,0)]
#             for k in range(LABEL_MAX_TRIES):
#                 d = base + k*step
#                 for dx,dy in dirs:
#                     cx = ax + dx*d; cy = ay + dy*d
#                     rect = (cx - w_cm/2, cx + w_cm/2, cy - h_cm/2, cy + h_cm/2)
#                     if all(not self._overlap_rect(rect, r) for r in taken_rects):
#                         taken_rects.append(rect); return cx, cy
#             cx = ax + base; cy = ay + base
#             rect = (cx - w_cm/2, cx + w_cm/2, cy - h_cm/2, cy + h_cm/2)
#             taken_rects.append(rect); return cx, cy

#         def draw_one(node, label_text, color):
#             if not node: return
#             x, y = _node_xy(node)
#             pw.addItem(pg.ScatterPlotItem(pos=[(x,y)], size=star_size_px,
#                                           brush=pg.mkBrush(color), pen=pg.mkPen("w", width=star_border_w),
#                                           symbol="star", antialias=True))
#             push_circle_rect(x, y, star_size_px)
#             tx, ty = place_bb(x, y, label_text, LABEL_BASE_PUDO, LABEL_STEP_PUDO)
#             ti = pg.TextItem(text=label_text, color=color, anchor=(0.5,0.5))
#             ti.setFont(self._make_font(label_px, bold=True, export=True))
#             ti.setPos(tx,ty); ti.setZValue(12); pw.addItem(ti)

#         for rec in self._map.order_meta:
#             oid = str(rec.get("id",""))
#             draw_one(rec.get("pickup_node"),  f"P{oid}", COLOR_PICKUP)
#             draw_one(rec.get("dropoff_node"), f"D{oid}", COLOR_DROPOFF)

#     # ---------------- 视域/导出 ----------------
#     def _global_bounds(self):
#         xs, ys = [], []
#         for n in self._map.nodes:
#             xs.append(float(n.position.x)); ys.append(float(n.position.y))
#         for props in self._world_nodes:
#             p = props.get("properties",{}) if isinstance(props,dict) else {}
#             poi = (p.get("poi_type") or p.get("type") or "").lower()
#             if poi not in BUILDING_TYPES: continue
#             loc = p.get("location",{}) or {}; bbox = p.get("bbox",{}) or {}
#             x = float(loc.get("x",0.0)); y = float(loc.get("y",0.0))
#             w = float(bbox.get("x",0.0)); h = float(bbox.get("y",0.0))
#             if w>0 and h>0: xs += [x-w/2, x+w/2]; ys += [y-h/2, y+h/2]
#         for path in self._bus_paths:
#             for x,y in path: xs.append(x); ys.append(y)
#         if not xs: return -1000, 1000, -1000, 1000
#         return min(xs)-GLOBAL_PAD_CM, max(xs)+GLOBAL_PAD_CM, min(ys)-GLOBAL_PAD_CM, max(ys)+GLOBAL_PAD_CM

#     def _local_bounds(self, reachable):
#         pts = []
#         for it in reachable.get("next_hop", []): pts.append((float(it["x"]), float(it["y"])))
#         for it in reachable.get("next_intersections", []): pts.append((float(it["x"]), float(it["y"])))
#         if self._agent_xy: pts.append(self._agent_xy)
#         if not pts: return -1000, 1000, -1000, 1000
#         xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
#         return min(xs)-LOCAL_MARGIN_CM, max(xs)+LOCAL_MARGIN_CM, min(ys)-LOCAL_MARGIN_CM, max(ys)+LOCAL_MARGIN_CM

#     def _export(self, pw: pg.PlotWidget, path: Optional[str], bounds,
#                 target_long_side_px=3200, return_bytes: bool=False) -> Optional[bytes]:
#         xmin, xmax, ymin, ymax = bounds
#         pw.plotItem.vb.setRange(xRange=(xmin, xmax), yRange=(ymin, ymax), padding=0)
#         QCoreApplication.processEvents()

#         span_x = max(1.0, xmax - xmin)
#         span_y = max(1.0, ymax - ymin)
#         aspect = span_x / span_y

#         if aspect >= 1.0:
#             width_px = int(target_long_side_px)
#         else:
#             width_px = int(target_long_side_px * aspect)

#         ex = ImageExporter(pw.plotItem)
#         ex.parameters()["width"] = max(800, width_px)
#         ex.parameters()["antialias"] = True

#         if return_bytes or path is None:
#             fd, tmp = tempfile.mkstemp(suffix=".png"); os.close(fd)
#             try:
#                 ex.export(tmp)
#                 with open(tmp, "rb") as f: data = f.read()
#             finally:
#                 try: os.remove(tmp)
#                 except Exception: pass
#             return data

#         ex.export(path)
#         return None

#     # ---------------- 交互放置（窗口） ----------------
#     def _place_label_live(self, anchor: Tuple[float,float], black_text: bool, base: float, step: float):
#         ax, ay = anchor; rlab = R_TEXT_BLACK if black_text else R_TEXT_COLOR
#         dirs = [(1,1),(-1,1),(1,-1),(-1,-1),(0,1),(0,-1),(1,0),(-1,0)]
#         for k in range(LABEL_MAX_TRIES):
#             d = base + k*step
#             for dx,dy in dirs:
#                 x = ax + dx*d; y = ay + dy*d
#                 if not self._collide_any(x,y,rlab):
#                     self._taken.append((x,y,rlab)); return x, y
#         self._taken.append((ax+base, ay+base, rlab))
#         return ax+base, ay+base

#     def _collide_any(self, x: float, y: float, r: float) -> bool:
#         for cx,cy,cr in self._taken:
#             if (x-cx)**2 + (y-cy)**2 < (r+cr)**2: return True
#         return False

#     def _add_text(self, text: str, x: float, y: float, color: str, target_plot: Optional[pg.PlotWidget]=None):
#         tp = target_plot or self.plot
#         ti = pg.TextItem(text=text, color=color, anchor=(0.5,0.5))
#         ti.setFont(self._label_font); ti.setPos(x,y); ti.setZValue(12)
#         tp.addItem(ti)

#     def _star_with_text(self, node, text: str, color: str, size_px: int=UI_STAR_SIZE, border_w: float=UI_STAR_BORDER_W):
#         if not node: return
#         x, y = _node_xy(node)
#         star = pg.ScatterPlotItem(pos=[(x,y)], size=size_px,
#                                   brush=pg.mkBrush(color), pen=pg.mkPen("w", width=border_w),
#                                   symbol="star", antialias=True)
#         star.setZValue(10); self.plot.addItem(star)
#         lx, ly = self._place_label_live((x,y), False, LABEL_BASE_PUDO, LABEL_STEP_PUDO)
#         self._add_text(text, lx, ly, color)

#     # ---------------- world 相关 ----------------
#     def _load_world(self, path: str):
#         try:
#             with open(path,"r",encoding="utf-8") as f: data = json.load(f)
#             self._world_nodes = data.get("nodes", [])
#             self._bus_paths = []
#             for r in data.get("bus_routes", []):
#                 pth = r.get("path", [])
#                 if len(pth) >= 2:
#                     pts = [(float(p.get("x",0))*100, float(p.get("y",0))*100) for p in pth]
#                     self._bus_paths.append(pts)
#         except Exception:
#             self._world_nodes = []; self._bus_paths = []

#     def _draw_bus_routes(self, target_plot, bus_width: float):
#         for pts in self._bus_paths:
#             xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
#             target_plot.addItem(pg.PlotDataItem(x=xs, y=ys,
#                                                 pen=pg.mkPen(COLOR_BUS_ROUTE, width=bus_width),
#                                                 antialias=True))

#     def _draw_buildings(self, target_plot):
#         """彩色：只画已知 typed 建筑块（restaurant/store/...）"""
#         if not self._world_nodes: return
#         for n in self._world_nodes:
#             props = n.get("properties", {}) or {}
#             poi   = (props.get("poi_type") or props.get("type") or "").strip().lower()
#             if poi not in BUILDING_TYPES: continue
#             loc  = props.get("location", {}) or {}
#             ori  = props.get("orientation", {}) or {}
#             bbox = props.get("bbox", {}) or {}
#             x = float(loc.get("x",0.0)); y = float(loc.get("y",0.0))
#             yaw = float(ori.get("yaw",0.0))
#             w = float(bbox.get("x",0.0)) or 600.0
#             h = float(bbox.get("y",0.0)) or 600.0

#             rect = QGraphicsRectItem(x-w/2.0, y-h/2.0, w, h)
#             rect.setTransformOriginPoint(x,y); rect.setRotation(yaw); rect.setZValue(-3)
#             rect.setPen(pg.mkPen("k", width=1))
#             rect.setBrush(pg.mkBrush(COLOR_BUILDING.get(poi, "#7F8C8D")))
#             target_plot.addItem(rect)

#             abbr = ABBR.get(poi,"?")
#             ti = pg.TextItem(text=abbr, color="w", anchor=(0.5,0.5))
#             f = QFont(self._bld_font); f.setPixelSize(UI_BUILDING_ABBR_PX); ti.setFont(f)
#             ti.setPos(x,y); ti.setZValue(-2); target_plot.addItem(ti)

#     def _draw_all_building_boxes(self, target_plot):
#         """灰色填充：只画普通 building（BP_Building_* 或 poi_type=='building'）"""
#         if not self._world_nodes: return
#         for n in self._world_nodes:
#             props = n.get("properties", {}) or {}
#             poi   = (props.get("poi_type") or props.get("type") or "").strip().lower()
#             inst  = str(n.get("instance_name", "") or "")
#             is_plain = (poi == "building") or inst.startswith("BP_Building")
#             if not is_plain: continue

#             loc  = props.get("location", {}) or {}
#             ori  = props.get("orientation", {}) or {}
#             bbox = props.get("bbox", {}) or {}
#             x = float(loc.get("x",0.0)); y = float(loc.get("y",0.0))
#             yaw = float(ori.get("yaw",0.0))
#             w = float(bbox.get("x",0.0)) or 600.0
#             h = float(bbox.get("y",0.0)) or 600.0

#             rect = QGraphicsRectItem(x-w/2.0, y-h/2.0, w, h)
#             rect.setTransformOriginPoint(x,y); rect.setRotation(yaw); rect.setZValue(-4)
#             pen = pg.mkPen(COLOR_PLAIN_BORDER, width=1.2)
#             fill_q = QColor(COLOR_BUILDING_PLAIN); fill_q.setAlpha(PLAIN_FILL_ALPHA)
#             rect.setPen(pen); rect.setBrush(pg.mkBrush(fill_q))
#             target_plot.addItem(rect)

#     # ---------------- 顶部按钮 ----------------
#     def _handle_reinit(self):
#         self.pause_animation(); self.reset_scene()
#         if self._on_reinit: self._on_reinit()

#     def _handle_save(self):
#         if self._last_reachable:
#             ts = time.strftime("%Y%m%d_%H%M%S")
#             self.save_images(self._last_reachable, f"global_{ts}.png", f"local_{ts}.png")

#     def _handle_go_clicked(self):
#         try:
#             tx = float(self.input_x.text().strip())
#             ty = float(self.input_y.text().strip())
#         except Exception:
#             self.info.append("[Go] invalid target xy"); return
#         if self._on_go: self._on_go(tx, ty)

#     # ---------------- 按坐标调试节点边 ----------------
#     def _clear_inspect_overlay(self):
#         if not self._inspect_items: return
#         for it in self._inspect_items:
#             try: self.plot.removeItem(it)
#             except Exception: pass
#         self._inspect_items.clear()

#     def _handle_inspect_clicked(self):
#         if self._map is None:
#             self.info.append("[Inspect] map is None"); return
#         try:
#             x = float(self.inspect_x.text().strip())
#             y = float(self.inspect_y.text().strip())
#             tol = float(self.inspect_tol.text().strip()) if self.inspect_tol.text().strip() else 200.0
#         except Exception:
#             self.info.append("[Inspect] invalid inputs"); return

#         nd = self._find_node_near(x, y, tol)
#         self._clear_inspect_overlay()

#         if nd is None:
#             self.info.append(f"[Inspect] no node within {tol:.0f} cm of ({int(x)}, {int(y)})"); return

#         nx, ny = _node_xy(nd)
#         node_item = pg.ScatterPlotItem(pos=[(nx, ny)], size=UI_NODE_SIZE*1.6,
#                                        brush=pg.mkBrush(COLOR_DEBUG_NODE),
#                                        pen=pg.mkPen("w", width=1.2),
#                                        symbol="o", antialias=True)
#         node_item.setZValue(15); self.plot.addItem(node_item); self._inspect_items.append(node_item)

#         lines: List[str] = []
#         tname = getattr(nd, "type", "")
#         lines.append(f"[Inspect] node at ({int(nx)}, {int(ny)}) type={tname} id={id(nd)}")

#         seen_nbs = set()
#         for v in self._map.adjacency_list.get(nd, []):
#             if v in seen_nbs: continue
#             seen_nbs.add(v)
#             vx, vy = _node_xy(v)
#             meta = {}
#             get_meta = getattr(self._map, "_get_edge_meta", None)
#             if callable(get_meta): meta = get_meta(nd, v) or {}
#             kind = (meta.get("kind") or ""); name = (meta.get("name") or "")
#             L = math.hypot(vx - nx, vy - ny); Lm = L / 100.0

#             edge_item = pg.PlotDataItem(x=[nx, vx], y=[ny, vy],
#                                         pen=pg.mkPen(COLOR_DEBUG_EDGE, width=DEBUG_EDGE_WIDTH),
#                                         antialias=True)
#             edge_item.setZValue(14); self.plot.addItem(edge_item); self._inspect_items.append(edge_item)

#             lines.append(f"  -> ({int(vx)}, {int(vy)})  len={Lm:.1f}m  kind='{kind}'  name='{name}'  nb_type={getattr(v,'type','')}")

#         xs = [nx]; ys = [ny]
#         for v in seen_nbs:
#             vx, vy = _node_xy(v); xs.append(vx); ys.append(vy)
#         if xs and ys:
#             pad = 800.0
#             xmin, xmax = min(xs)-pad, max(xs)+pad
#             ymin, ymax = min(ys)-pad, max(ys)+pad
#             self.plot.plotItem.vb.setRange(xRange=(xmin, xmax), yRange=(ymin, ymax), padding=0)

#         self.info.append("\n".join(lines))

#     def _find_node_near(self, x: float, y: float, tol_cm: float = 200.0):
#         """在 map.nodes 中找与 (x,y) 最近且距离 <= tol_cm 的节点；否则返回 None。"""
#         if self._map is None: return None
#         best = None; best_d2 = None
#         tx, ty = float(x), float(y); thr2 = float(tol_cm) * float(tol_cm)
#         for n in self._map.nodes:
#             nx, ny = _node_xy(n)
#             dx, dy = nx - tx, ny - ty
#             d2 = dx*dx + dy*dy
#             if d2 <= thr2 and (best is None or d2 < (best_d2 or 1e99) - 1e-9):
#                 best = n; best_d2 = d2
#         return best
