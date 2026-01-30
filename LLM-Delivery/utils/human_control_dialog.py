#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Human control dialog
Human control interface integrated into MapObserver
"""

import sys
import os
import json
import math
from typing import List, Optional, Dict, Any
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QTextEdit, QGroupBox,
    QListWidget, QListWidgetItem, QMessageBox, QSplitter,
    QWidget
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QColor

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from Base.DeliveryMan import DMAction, DMActionKind
from utils.map_observer import OrdersDialog  # already available on your side


class HumanControlDialog(QDialog):
    """Human control dialog"""

    def __init__(self, parent=None, agents: List[Any] = None):
        super().__init__(parent)
        self.agents = agents or []
        self.target_agent = None
        self._last_order_signature = None  # 用于检测订单集合变化

        # === 新增：按位置刷新用 ===
        self._last_xy_cm: Optional[tuple] = None
        self._poi_refresh_eps_cm: float = 20.0  # 移动超过 20cm 触发刷新

        # === 新增：订单文本缓存，避免不必要重绘引发回到顶部 ===
        self._orders_text_cache: str = ""

        # Orders 窗口引用/工厂/数据源（可选）
        self._orders_dialog: Optional[QDialog] = None
        self._orders_dialog_factory = None   # callable: () -> OrdersDialog
        self._order_manager = None
        self._clock = None

        self.setup_ui()
        self.setup_timer()

    # ---------- OrdersDialog 注入 / 打开 ----------
    def set_orders_dialog(self, dlg: QDialog):
        self._orders_dialog = dlg

    def set_orders_dialog_factory(self, factory_callable):
        self._orders_dialog_factory = factory_callable

    def set_orders_sources(self, order_manager=None, clock=None):
        if order_manager is not None:
            self._order_manager = order_manager
        if clock is not None:
            self._clock = clock

    def _open_orders_dialog(self):
        if self._orders_dialog is None:
            if callable(self._orders_dialog_factory):
                self._orders_dialog = self._orders_dialog_factory()
            elif OrdersDialog is not None:
                try:
                    self._orders_dialog = OrdersDialog(
                        parent=self,
                        order_manager=self._order_manager,
                        agents=self.agents,
                        clock=self._clock
                    )
                except Exception:
                    self._orders_dialog = OrdersDialog(parent=self)
            else:
                QMessageBox.warning(self, "提示", "Orders 窗口不可用：未设置工厂函数且无法导入 OrdersDialog")
                return

        try:
            if hasattr(self._orders_dialog, "set_sources"):
                self._orders_dialog.set_sources(
                    order_manager=self._order_manager,
                    agents=self.agents,
                    clock=self._clock
                )
            if hasattr(self._orders_dialog, "refresh_all"):
                self._orders_dialog.refresh_all()
            elif hasattr(self._orders_dialog, "btn_refresh"):
                self._orders_dialog.btn_refresh.click()
        except Exception:
            pass

        self._orders_dialog.show()
        self._orders_dialog.raise_()
        self._orders_dialog.activateWindow()

    # ---------- UI ----------
    def setup_ui(self):
        self.setWindowTitle("Human Control - DeliveryMan")
        self.setGeometry(100, 100, 1200, 760)

        main_layout = QVBoxLayout(self)
        splitter = QSplitter()
        main_layout.addWidget(splitter)

        left_panel = self.create_left_panel()
        right_panel = self.create_right_panel()
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)

        splitter.setSizes([560, 440])
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

    def create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Agent selection
        agent_group = QGroupBox("Select Agent")
        agent_layout = QVBoxLayout(agent_group)
        self.agent_combo = QComboBox()
        self.agent_combo.currentTextChanged.connect(self.on_agent_changed)
        agent_layout.addWidget(self.agent_combo)
        for agent in self.agents:
            self.agent_combo.addItem(f"Agent {agent.agent_id}")
        layout.addWidget(agent_group)

        # Status info
        status_group = QGroupBox("Current Status")
        status_layout = QVBoxLayout(status_group)
        self.status_labels = {}
        for label_text, key in [
            ("Position", "position"), ("Mode", "mode"), ("Energy", "energy"),
            ("Earnings", "earnings"), ("Sim Time", "sim_time"),
            ("Active Orders", "active_orders"), ("Carrying", "carrying"),
            ("Current Action", "current_action"), ("Control Mode", "human_control_mode")
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{label_text}:"))
            val = QLabel("N/A")
            val.setStyleSheet("font-weight: bold; color: #2E8B57;")
            row.addWidget(val)
            self.status_labels[key] = val
            status_layout.addLayout(row)
        layout.addWidget(status_group)

        # My orders (QTextEdit: scrollable, monospace, no wrap)
        orders_group = QGroupBox("My Orders (active orders)")
        orders_layout = QVBoxLayout(orders_group)
        self.orders_text = QTextEdit()
        self.orders_text.setReadOnly(True)
        self.orders_text.setLineWrapMode(QTextEdit.NoWrap)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(9)
        self.orders_text.setFont(mono)
        self.orders_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.orders_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.orders_text.setMinimumWidth(520)
        self.orders_text.setMinimumHeight(140)
        orders_layout.addWidget(self.orders_text)
        layout.addWidget(orders_group)

        # Control mode
        control_group = QGroupBox("Control Mode")
        control_layout = QVBoxLayout(control_group)
        self.mode_label = QLabel("Current mode: VLM control")
        self.mode_label.setStyleSheet("font-weight: bold; color: #FF6347;")
        control_layout.addWidget(self.mode_label)
        self.toggle_mode_btn = QPushButton("Switch to Human Control")
        self.toggle_mode_btn.clicked.connect(self.toggle_control_mode)
        control_layout.addWidget(self.toggle_mode_btn)
        layout.addWidget(control_group)

        # Available POIs
        actions_group = QGroupBox("Available POIs")
        actions_layout = QVBoxLayout(actions_group)
        self.actions_list = QListWidget()
        self.actions_list.setTextElideMode(Qt.ElideNone)
        self.actions_list.setWordWrap(False)
        self.actions_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.actions_list.setHorizontalScrollMode(self.actions_list.ScrollPerPixel)
        self.actions_list.setMinimumWidth(520)
        self.actions_list.itemClicked.connect(self.on_poi_selected)
        actions_layout.addWidget(self.actions_list)
        self.populate_poi_list()
        layout.addWidget(actions_group)

        layout.setStretchFactor(actions_group, 1)
        return panel

    def create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Action input
        input_group = QGroupBox("Action Input")
        input_layout = QVBoxLayout(input_group)
        row = QHBoxLayout()
        row.addWidget(QLabel("Action Type:"))
        self.action_combo = QComboBox()
        self.action_combo.addItems([action.value for action in DMActionKind])
        self.action_combo.currentTextChanged.connect(self.on_action_type_changed)
        row.addWidget(self.action_combo)
        input_layout.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Parameters (JSON):"))
        self.params_input = QLineEdit()
        self.params_input.setPlaceholderText('e.g.: {"tx": 100.0, "ty": 200.0}')
        row2.addWidget(self.params_input)
        input_layout.addLayout(row2)

        btn_row = QHBoxLayout()
        self.submit_btn = QPushButton("Submit Action")
        self.submit_btn.clicked.connect(self.submit_action)
        self.submit_btn.setEnabled(False)
        btn_row.addWidget(self.submit_btn)
        self.clear_btn = QPushButton("Clear Input")
        self.clear_btn.clicked.connect(self.clear_input)
        btn_row.addWidget(self.clear_btn)
        input_layout.addLayout(btn_row)
        layout.addWidget(input_group)

        # Action queue
        queue_group = QGroupBox("Action Queue")
        queue_layout = QVBoxLayout(queue_group)
        self.queue_list = QListWidget()
        queue_layout.addWidget(self.queue_list)
        btns = QHBoxLayout()
        self.clear_queue_btn = QPushButton("Clear Queue")
        self.clear_queue_btn.clicked.connect(self.clear_queue)
        btns.addWidget(self.clear_queue_btn)
        queue_layout.addLayout(btns)
        layout.addWidget(queue_group)

        # Logs
        log_group = QGroupBox("System Logs")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(220)
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)

        return panel

    def setup_timer(self):
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_status)
        self.update_timer.start(1000)
        self.last_action_count = 0
        self.last_current_action = None

    # ---------- 辅助 ----------
    def _poi_anchor_xy(self, meta: Dict[str, Any]) -> Optional[tuple]:
        try:
            node = meta.get("node")
            ptype = (getattr(node, "type", "") or "").strip().lower() if node else ""
            if ptype in ("charging_station", "bus_station"):
                if node and hasattr(node, "position"):
                    return float(node.position.x), float(node.position.y)
                dock = meta.get("dock_node")
                if dock and hasattr(dock, "position"):
                    return float(dock.position.x), float(dock.position.y)
                ctr = meta.get("center")
                if isinstance(ctr, (list, tuple)) and len(ctr) >= 2:
                    return float(ctr[0]), float(ctr[1])
                return None
            door = meta.get("door_node")
            if door and hasattr(door, "position"):
                return float(door.position.x), float(door.position.y)
            dock = meta.get("dock_node")
            if dock and hasattr(dock, "position"):
                return float(dock.position.x), float(dock.position.y)
            if node and hasattr(node, "position"):
                return float(node.position.x), float(node.position.y)
            ctr = meta.get("center")
            if isinstance(ctr, (list, tuple)) and len(ctr) >= 2:
                return float(ctr[0]), float(ctr[1])
        except Exception:
            pass
        return None

    def _order_endpoints(self, o):
        def _safe_xy(node):
            try:
                return float(node.position.x), float(node.position.y)
            except Exception:
                return None
        pu_node = getattr(o, "pickup_node", None)
        do_node = getattr(o, "dropoff_node", None)
        pu_xy = _safe_xy(pu_node)
        do_xy = _safe_xy(do_node)
        pu_road = getattr(o, "pickup_road_name", "") or ""
        do_road = getattr(o, "dropoff_road_name", "") or ""
        return (pu_xy, pu_road), (do_xy, do_road)

    def _current_order_signature(self):
        """(Tuple of active order IDs, tuple of help order IDs) — used to detect changes"""
        try:
            active_ids = tuple(sorted(
                int(getattr(o, "id", -1))
                for o in (getattr(self.target_agent, "active_orders", []) or [])
                if getattr(o, "id", None) is not None
            ))
        except Exception:
            active_ids = tuple()
        try:
            help_dict = getattr(self.target_agent, "help_orders", {}) or {}
            help_ids = tuple(sorted(int(k) for k in help_dict.keys())) if isinstance(help_dict, dict) else tuple()
        except Exception:
            help_ids = tuple()
        return (active_ids, help_ids)

    # ---------- 事件 ----------
    def set_agents(self, agents: List[Any]):
        self.agents = agents
        self.agent_combo.clear()
        for agent in self.agents:
            self.agent_combo.addItem(f"Agent {agent.agent_id}")
        if self.agents:
            self.target_agent = self.agents[0]
            if hasattr(self, 'update_timer') and self.update_timer:
                self.update_timer.timeout.emit()
                self.populate_poi_list()

    def on_agent_changed(self, agent_text):
        if not self.agents:
            return
        try:
            agent_id = agent_text.split()[-1]
            for agent in self.agents:
                if agent.agent_id == agent_id:
                    self.target_agent = agent
                    self.populate_poi_list()
                    self.log_message(f"🎯 Selected Agent {agent_id}")
                    self.log_message(f"   Position: ({agent.x:.1f}, {agent.y:.1f})")
                    self.log_message(f"   Mode: {agent.mode.value}")
                    self.log_message(f"   Control Mode: {'Human Control' if agent.human_control_mode else 'VLM Control'}")
                    break
        except Exception as e:
            self.log_message(f"Failed to select agent: {e}")

    def update_status(self):
        if not self.target_agent or not self.status_labels:
            return
        try:
            status = self.target_agent.get_current_status()
            self.status_labels["position"].setText(f"({status['position'][0]:.1f}, {status['position'][1]:.1f})")
            self.status_labels["mode"].setText(status['mode'])
            self.status_labels["energy"].setText(f"{status['energy']:.1f}%")
            self.status_labels["earnings"].setText(f"${status['earnings']:.2f}")
            sim_s = float(status.get('sim_time_s', 0.0) or 0.0)
            h = int(sim_s // 3600); m = int((sim_s % 3600) // 60); s = int(sim_s % 60)
            self.status_labels["sim_time"].setText(f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}")
            self.status_labels["active_orders"].setText(str(len(status.get('active_orders', []) or [])))
            self.status_labels["carrying"].setText(str(len(status['carrying'])))
            self.status_labels["current_action"].setText(status['current_action'] or "None")

            if status['human_control_mode']:
                self.mode_label.setText("Current mode: Human control")
                self.mode_label.setStyleSheet("font-weight: bold; color: #32CD32;")
                self.toggle_mode_btn.setText("Switch to VLM Control")
                self.submit_btn.setEnabled(True)
            else:
                self.mode_label.setText("Current mode: VLM control")
                self.mode_label.setStyleSheet("font-weight: bold; color: #FF6347;")
                self.toggle_mode_btn.setText("Switch to Human Control")
                self.submit_btn.setEnabled(False)

            self.update_queue_display()
            self.update_orders_display(status)
            self.monitor_action_execution(status)

            # 订单集合变化 -> 刷新 POI 列表
            sig = self._current_order_signature()
            if sig != self._last_order_signature:
                self._last_order_signature = sig
                self.populate_poi_list()

            # === 位置变化 -> 刷新 POI/订单端点距离与排序 ===
            try:
                curx = float(getattr(self.target_agent, "x", 0.0))
                cury = float(getattr(self.target_agent, "y", 0.0))
                if self._last_xy_cm is None:
                    self._last_xy_cm = (curx, cury)
                else:
                    dx = curx - self._last_xy_cm[0]
                    dy = cury - self._last_xy_cm[1]
                    if (dx*dx + dy*dy) ** 0.5 >= float(self._poi_refresh_eps_cm):
                        self._last_xy_cm = (curx, cury)
                        self.populate_poi_list()
            except Exception:
                pass

        except Exception as e:
            self.log_message(f"Error while updating status: {e}")

    def update_queue_display(self):
        if not self.queue_list:
            return
        self.queue_list.clear()
        if self.target_agent and hasattr(self.target_agent, 'human_action_queue'):
            for i, action in enumerate(self.target_agent.human_action_queue):
                txt = f"{i+1}. {action.kind.value}"
                if action.data:
                    txt += f" - {action.data}"
                self.queue_list.addItem(txt)

    # === 扩展：显示订单 + 食物 items（温度/破损/气味污染） ===
    def _order_to_text_safe(self, o) -> str:
        """
        优先使用 o.to_text() 作为“头部”，随后安全地追加 Items 与 Note 段。
        若头部已含有 Items/Note（不区分大小写），则避免重复。
        """
        # 1) 头部：尽量沿用 o.to_text()
        header = None
        try:
            if hasattr(o, "to_text"):
                t = o.to_text()
                if isinstance(t, str) and t.strip():
                    header = t.rstrip("\n")
        except Exception:
            header = None

        # 2) If there is no header, construct a concise one
        if not header:
            oid = getattr(o, 'id', '?')
            pu = getattr(o, 'pickup_road_name', '') or ''
            do = getattr(o, 'dropoff_road_name', '') or ''
            picked = getattr(o, 'has_picked_up', False)
            delivered = getattr(o, 'has_delivered', False)
            earn = getattr(o, 'earnings', None)
            earn_str = f"${float(earn):.2f}" if isinstance(earn, (int, float)) else "N/A"
            header_lines = [
                f"[Order #{oid}]",
                f"  Route  : {pu} -> {do}",
                f"  Status : picked={picked}  delivered={delivered}",
                f"  $$     : {earn_str}",
            ]
            header = "\n".join(header_lines)

        # 3) Normalize header text (for easier contains checks)
        header_lc = header.lower()

        # 4) Assemble final text: header + Items (if needed) + Note (if needed)
        lines = [header]

        # ---- Items ----
        try:
            items = list(getattr(o, "items", []) or [])
        except Exception:
            items = []

        if items and ("items" not in header_lc):  # Add Items only if header doesn't already contain them
            lines.append("  Items  :")
            for it in items:
                name = getattr(it, "name", str(it))

                # Support both temp_c / serving_temp_c
                t_val = getattr(it, "temp_c", None)
                if t_val is None or (isinstance(t_val, float) and t_val != t_val):  # NaN 检查：t!=t
                    t_val = getattr(it, "serving_temp_c", None)
                try:
                    t = float(t_val) if t_val is not None else float("nan")
                except Exception:
                    t = float("nan")
                t_str = f"{t:.1f}°C" if (t == t) else "N/A"

                oc = getattr(it, "odor_contamination", None)
                try:
                    oc_val = max(0.0, min(1.0, float(oc)))
                    contam_str = f"{oc_val*100:.0f}%"
                except (TypeError, ValueError):
                    contam_str = "N/A"

                try:
                    damage_level = int(getattr(it, "damage_level", 0))
                    damage_str = str(damage_level) if (0 <= damage_level <= 3) else "N/A"
                except Exception:
                    damage_str = "N/A"

                lines.append(f"    - {name}  (T={t_str}, damage={damage_str}, odor contam={contam_str})")

        # ---- Note ----
        note = getattr(o, "special_note", "")
        if isinstance(note, str) and note.strip() and ("note" not in header_lc):
            lines.append(f"  Note   : {note.strip()}")

        # 5) Return and ensure ending with newline
        out = "\n".join(lines)
        if not out.endswith("\n"):
            out += "\n"
        return out


    def update_orders_display(self, status: Dict[str, Any]):
        """
        渲染 active + help orders 到 QTextEdit。
        - 仅当内容变更时才写入，避免把滚动条跳回顶部；
        - 在写入前后保存/恢复滚动位置。
        """
        try:
            active = list(getattr(self.target_agent, "active_orders", []) or [])
            help_dict = getattr(self.target_agent, "help_orders", {}) or {}
            help_orders = list(help_dict.values()) if isinstance(help_dict, dict) else []
            order_list = active + help_orders

            lines: List[str] = [f"Active count: {len(order_list)}\n"]
            for o in order_list:
                txt = self._order_to_text_safe(o)
                lines.append(txt if txt.endswith("\n") else (txt + "\n"))

            new_text = "".join(lines) if lines else "Active count: 0\n"

            # —— 如果内容没变，直接返回，不触碰滚动条
            if new_text == self._orders_text_cache:
                return

            # —— 记录当前滚动位置（绝对值 + 比例双保险）
            vbar = self.orders_text.verticalScrollBar()
            hbar = self.orders_text.horizontalScrollBar()
            v_val = vbar.value()
            h_val = hbar.value()
            v_max = max(1, vbar.maximum())
            h_max = max(1, hbar.maximum())
            v_ratio = v_val / float(v_max)
            h_ratio = h_val / float(h_max)

            # 写入文本
            self.orders_text.setPlainText(new_text)
            self._orders_text_cache = new_text

            # —— 恢复滚动位置（先按绝对值，若范围变化太大则按比例）
            try:
                vbar2 = self.orders_text.verticalScrollBar()
                hbar2 = self.orders_text.horizontalScrollBar()

                new_v_max = max(1, vbar2.maximum())
                new_h_max = max(1, hbar2.maximum())

                # 尝试原值
                vbar2.setValue(min(v_val, new_v_max))
                hbar2.setValue(min(h_val, new_h_max))

                # 如果明显偏差（例如由重绘导致范围变化），再按比例兜底
                if abs(vbar2.value() - v_val) > 5 and new_v_max > 1:
                    vbar2.setValue(int(v_ratio * new_v_max))
                if abs(hbar2.value() - h_val) > 5 and new_h_max > 1:
                    hbar2.setValue(int(h_ratio * new_h_max))
            except Exception:
                pass

        except Exception as e:
            self.orders_text.setPlainText(f"[显示订单时出错]: {e}")

    def monitor_action_execution(self, status):
        if not self.target_agent:
            return
        try:
            current_action = status.get('current_action')
            queue_count = len(self.target_agent.human_action_queue) if hasattr(self.target_agent, 'human_action_queue') else 0
            if current_action and current_action != self.last_current_action:
                self.log_message(f"🚀 Started executing action: {current_action}")
                self.last_current_action = current_action
            if queue_count != self.last_action_count:
                if queue_count < self.last_action_count:
                    self.log_message(f"✅ Action removed from queue, remaining length: {queue_count}")
                self.last_action_count = queue_count
            if self.last_current_action and not current_action:
                self.log_message(f"✅ Action completed: {self.last_current_action}")
                self.last_current_action = None
        except Exception as e:
            self.log_message(f"Error while monitoring action execution: {e}")

    def toggle_control_mode(self):
        if not self.target_agent:
            QMessageBox.warning(self, "警告", "没有选择目标Agent")
            return
        try:
            return  # 暂不切换
        except Exception as e:
            QMessageBox.critical(self, "错误", f"切换控制模式失败: {e}")

    def on_action_selected(self, item):
        action_type = item.data(256)
        self.action_combo.setCurrentText(action_type)
        self.on_action_type_changed(action_type)

    def on_action_type_changed(self, action_type):
        defaults = {
            "move_to": '{"tx": 100.0, "ty": 200.0}',
            "rest": '{"target_pct": 100.0}',
            "wait": '{"duration_s": 30.0}',
            "charge_escooter": '{"target_pct": 100.0}',
            "buy": '{"item_id": "energy_drink", "qty": 1}',
            "say": '{"text": "Hello!"}',
            "turn_around": '{"angle": 180.0}',
            "view_orders": "{}",
            # 接单：既支持单个，也支持多个
            # - 单个:  {"oid": 12}
            # - 多个:  {"oids": [12,18]}  或  {"ids": [12,18]}
            "accept_order": '{"oids": [12, 18]}',

            # 放入保温袋
            "place_food_in_bag": '{"bag_cmd": "order 12: 1,2 -> A; 3 -> B"}',

            # 运输方式切换（等价于 SWITCH(to="e-scooter")）
            "switch_transport": '{"to": "e-scooter"}',

            # 投递：method 可取 "leave_at_door" | "hand_to_customer" | "knock" | "call"
            "drop_off": '{"oid": 12, "method": "leave_at_door"}',
        }

        self.params_input.setText(defaults.get(action_type, "{}"))

    def submit_action(self):
        if not self.target_agent:
            QMessageBox.warning(self, "Warning", "No target agent selected")
            return
        if not self.target_agent.human_control_mode:
            QMessageBox.warning(self, "Warning", "Current mode is not human control")
            return
        try:
            action_type = self.action_combo.currentText()
            params_text = self.params_input.text().strip()

            if action_type.lower() == "view_orders":
                self._open_orders_dialog()

            params = json.loads(params_text) if params_text else {}
            action = self.target_agent.create_human_action(action_type, **params)
            success = self.target_agent.submit_human_action(action)
            if success:
                self.log_message(f"📤 Submitted action: {action_type}")
                if params:
                    self.log_message(f"   Params: {params}")
                self.log_message(f"   Queue position: {len(self.target_agent.human_action_queue)}")
                self.clear_input()
                if any(k in action_type.lower() for k in ("accept_order", "pickup", "drop_off", "accept_help")):
                    QTimer.singleShot(200, self.populate_poi_list)
            else:
                self.log_message(f"❌ Failed to submit action: {action_type}")
                self.log_message("   Possible reason: not in human control mode or invalid params")
        except json.JSONDecodeError:
            QMessageBox.warning(self, "Error", "Invalid parameter format, please input valid JSON")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to submit action: {e}")

    def clear_input(self):
        self.params_input.clear()

    def clear_queue(self):
        if self.target_agent and hasattr(self.target_agent, 'human_action_queue'):
            n = len(self.target_agent.human_action_queue)
            self.target_agent.human_action_queue.clear()
            self.log_message(f"🗑️ 已清空动作队列 (清除了 {n} 个动作)")

    def log_message(self, message):
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        if hasattr(self, 'log_text') and self.log_text is not None:
            self.log_text.append(f"[{ts}] {message}")
        else:
            print(f"[{ts}] {message}")

    # ---------- POI & 订单端点 ----------
    def _fmt_m(self, cm_val: float) -> str:
        try:
            m = float(cm_val) / 100.0
            return f"{m:.1f}m"
        except Exception:
            return "N/A"

    def populate_poi_list(self):
        """先放订单端点（带最短路距离）；再放地图 POI（按最短路距离排序并显示距离）。"""
        self.actions_list.clear()

        # ===== “我的订单端点”（最上方，带距离）=====
        if self.target_agent:
            active = list(getattr(self.target_agent, "active_orders", []) or [])
            help_dict = getattr(self.target_agent, "help_orders", {}) or {}
            help_orders = list(help_dict.values()) if isinstance(help_dict, dict) else []
            order_list = active + help_orders

            if order_list:
                header = QListWidgetItem("—— 我的订单 ——")
                header.setFlags(header.flags() & ~Qt.ItemIsSelectable)
                header.setForeground(QColor("#666"))
                self.actions_list.addItem(header)

            sx = float(getattr(self.target_agent, "x", 0.0))
            sy = float(getattr(self.target_agent, "y", 0.0))
            city_map = getattr(self.target_agent, "city_map", None)

            for o in order_list:
                oid = getattr(o, "id", None)
                pu_road = getattr(o, "pickup_road_name", "") or ""
                do_road = getattr(o, "dropoff_road_name", "") or ""
                pu_node = getattr(o, "pickup_node", None)
                do_node = getattr(o, "dropoff_node", None)

                def _safe_xy(nd):
                    try:
                        return float(nd.position.x), float(nd.position.y)
                    except Exception:
                        return None
                pu_xy = _safe_xy(pu_node)
                do_xy = _safe_xy(do_node)

                def _dist_to_node(nd, xy_fallback):
                    if nd is None:
                        return None
                    try:
                        if city_map and hasattr(city_map, "shortest_path_xy_to_node"):
                            _path, dist_cm, _ = city_map.shortest_path_xy_to_node(sx, sy, nd)
                            if isinstance(dist_cm, (int, float)) and math.isfinite(dist_cm):
                                return float(dist_cm)
                        if xy_fallback:
                            dx = xy_fallback[0] - sx
                            dy = xy_fallback[1] - sy
                            return (dx * dx + dy * dy) ** 0.5
                    except Exception:
                        if xy_fallback:
                            dx = xy_fallback[0] - sx
                            dy = xy_fallback[1] - sy
                            return (dx * dx + dy * dy) ** 0.5
                    return None

                if pu_xy:
                    dist_cm = _dist_to_node(pu_node, pu_xy)
                    dist_txt = f"  • {self._fmt_m(dist_cm)}" if dist_cm is not None else ""
                    txt = f"Order #{oid} • Pickup"
                    if pu_road:
                        txt += f"  ({pu_road})"
                    txt += dist_txt

                    it_pu = QListWidgetItem(txt)
                    it_pu.setData(256, {
                        "name": f"Order #{oid} Pickup",
                        "ptype": "order_pickup",
                        "road_name": pu_road,
                        "xy": pu_xy,
                        "order_id": oid,
                        "role": "pickup",
                        "dist_cm": dist_cm,
                    })
                    it_pu.setForeground(QColor("#2a7fff"))
                    self.actions_list.addItem(it_pu)

                if do_xy:
                    dist_cm = _dist_to_node(do_node, do_xy)
                    dist_txt = f"  • {self._fmt_m(dist_cm)}" if dist_cm is not None else ""
                    txt = f"Order #{oid} • Dropoff"
                    if do_road:
                        txt += f"  ({do_road})"
                    txt += dist_txt

                    it_do = QListWidgetItem(txt)
                    it_do.setData(256, {
                        "name": f"Order #{oid} Dropoff",
                        "ptype": "order_dropoff",
                        "road_name": do_road,
                        "xy": do_xy,
                        "order_id": oid,
                        "role": "dropoff",
                        "dist_cm": dist_cm,
                    })
                    it_do.setForeground(QColor("#22aa44"))
                    self.actions_list.addItem(it_do)

        # ===== 地图 POI（按最短路距离排序并显示距离）=====
        if self.target_agent and hasattr(self.target_agent, "city_map"):
            try:
                city_map = getattr(self.target_agent, "city_map", None)
                sx = float(getattr(self.target_agent, "x", 0.0))
                sy = float(getattr(self.target_agent, "y", 0.0))
                cand_list = city_map.list_direct_reachable_pois_xy(sx, sy) if city_map else []
            except Exception:
                cand_list = []

            if cand_list:
                header = QListWidgetItem("—— 地图 POI（按最短路距离） ——")
                header.setFlags(header.flags() & ~Qt.ItemIsSelectable)
                header.setForeground(QColor("#666"))
                self.actions_list.addItem(header)

            for p in cand_list:
                name = p.get("name") or (p.get("type") or "POI")
                rn = p.get("road_name") or ""
                dist_txt = self._fmt_m(p.get("dist_cm", 0.0))
                subtitle = (p.get("type") or "")
                if rn:
                    subtitle += f" @ {rn}"
                txt = f"{name}  ({subtitle})  • {dist_txt}"
                it = QListWidgetItem(txt)
                xy = (p.get("x"), p.get("y"))
                it.setData(256, {"name": name, "ptype": p.get("type"), "road_name": rn, "xy": xy})
                self.actions_list.addItem(it)

        if self.actions_list.count() == 0:
            it = QListWidgetItem("（暂无可用 POI 或订单端点）")
            it.setFlags(it.flags() & ~Qt.ItemIsSelectable)
            it.setForeground(QColor("#999"))
            self.actions_list.addItem(it)

    def on_poi_selected(self, item: QListWidgetItem):
        """选择一个 POI 后，自动构造 move_to 动作参数（坐标单位 cm）。"""
        data = item.data(256) or {}
        xy = data.get("xy")
        if not xy:
            self.log_message("所选 POI 缺少有效坐标，无法生成移动目标")
            return
        tx, ty = float(xy[0]), float(xy[1])
        self.action_combo.setCurrentText("move_to")
        self.params_input.setText(json.dumps({"tx": tx, "ty": ty}))
        self.submit_btn.setEnabled(bool(self.target_agent and self.target_agent.human_control_mode))
        self.log_message(f"🎯 已选择 POI: {data.get('name') or ''} -> 目标({tx:.1f}, {ty:.1f})")
