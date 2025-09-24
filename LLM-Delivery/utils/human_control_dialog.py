#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人类控制对话框
集成到MapObserver中的人类控制界面
"""

import sys
import os
import json
from typing import List, Optional, Dict, Any
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QComboBox, QLineEdit, QTextEdit, QGroupBox, QGridLayout,
    QListWidget, QListWidgetItem, QMessageBox, QSplitter,
    QTabWidget, QWidget, QFrame
)
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from Base.DeliveryMan import DMAction, DMActionKind

class HumanControlDialog(QDialog):
    """人类控制对话框"""
    
    def __init__(self, parent=None, agents: List[Any] = None):
        super().__init__(parent)
        self.agents = agents or []
        self.target_agent = None
        self.setup_ui()
        self.setup_timer()
        
    def setup_ui(self):
        """设置UI界面"""
        self.setWindowTitle("人类控制界面 - DeliveryMan")
        self.setGeometry(100, 100, 1000, 700)
        
        # 主布局
        main_layout = QVBoxLayout(self)
        
        # 创建分割器
        splitter = QSplitter()
        main_layout.addWidget(splitter)
        
        # 左侧：状态、订单和控制面板
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # 右侧：动作输入、队列和日志
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # 设置分割器比例
        splitter.setSizes([400, 600])
        
    def create_left_panel(self):
        """创建左侧状态面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Agent选择组
        agent_group = QGroupBox("选择Agent")
        agent_layout = QVBoxLayout(agent_group)
        
        self.agent_combo = QComboBox()
        self.agent_combo.currentTextChanged.connect(self.on_agent_changed)
        agent_layout.addWidget(self.agent_combo)
        
        # 填充Agent列表
        for agent in self.agents:
            self.agent_combo.addItem(f"Agent {agent.agent_id}")
            
        layout.addWidget(agent_group)
        
        # 状态信息组
        status_group = QGroupBox("当前状态")
        status_layout = QVBoxLayout(status_group)
        
        self.status_labels = {}
        status_items = [
            ("位置", "position"),
            ("模式", "mode"), 
            ("能量", "energy"),
            ("收入", "earnings"),
            ("模拟时间", "sim_time"),
            ("活跃订单", "active_orders"),
            ("携带物品", "carrying"),
            ("当前动作", "current_action"),
            ("控制模式", "human_control_mode")
        ]
        
        for label_text, key in status_items:
            row_layout = QHBoxLayout()
            row_layout.addWidget(QLabel(f"{label_text}:"))
            value_label = QLabel("N/A")
            value_label.setStyleSheet("font-weight: bold; color: #2E8B57;")
            row_layout.addWidget(value_label)
            self.status_labels[key] = value_label
            status_layout.addLayout(row_layout)
            
        layout.addWidget(status_group)

        # 我的订单（放左侧，保证可见）
        orders_group = QGroupBox("我的订单")
        orders_layout = QVBoxLayout(orders_group)
        self.orders_list = QListWidget()
        orders_layout.addWidget(self.orders_list)
        layout.addWidget(orders_group)
        
        # 控制模式切换
        control_group = QGroupBox("控制模式")
        control_layout = QVBoxLayout(control_group)
        
        self.mode_label = QLabel("当前模式: VLM控制")
        self.mode_label.setStyleSheet("font-weight: bold; color: #FF6347;")
        control_layout.addWidget(self.mode_label)
        
        self.toggle_mode_btn = QPushButton("切换到人类控制")
        self.toggle_mode_btn.clicked.connect(self.toggle_control_mode)
        control_layout.addWidget(self.toggle_mode_btn)
        
        layout.addWidget(control_group)
        
        # 可选POI列表
        actions_group = QGroupBox("可选POI")
        actions_layout = QVBoxLayout(actions_group)
        
        self.actions_list = QListWidget()  # 复用现有变量名以减少改动
        self.actions_list.itemClicked.connect(self.on_poi_selected)
        actions_layout.addWidget(self.actions_list)

        # 初始化填充 POI
        self.populate_poi_list()

        layout.addWidget(actions_group)
        return panel
        
    def create_right_panel(self):
        """创建右侧输入面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 动作输入组
        input_group = QGroupBox("动作输入")
        input_layout = QVBoxLayout(input_group)
        
        # 动作类型选择
        action_row = QHBoxLayout()
        action_row.addWidget(QLabel("动作类型:"))
        self.action_combo = QComboBox()
        self.action_combo.addItems([action.value for action in DMActionKind])
        self.action_combo.currentTextChanged.connect(self.on_action_type_changed)
        action_row.addWidget(self.action_combo)
        input_layout.addLayout(action_row)
        
        # 参数输入
        params_row = QHBoxLayout()
        params_row.addWidget(QLabel("参数 (JSON格式):"))
        self.params_input = QLineEdit()
        self.params_input.setPlaceholderText('例如: {"tx": 100.0, "ty": 200.0}')
        params_row.addWidget(self.params_input)
        input_layout.addLayout(params_row)
        
        # 按钮行
        button_row = QHBoxLayout()
        self.submit_btn = QPushButton("提交动作")
        self.submit_btn.clicked.connect(self.submit_action)
        self.submit_btn.setEnabled(False)
        button_row.addWidget(self.submit_btn)
        
        self.clear_btn = QPushButton("清空输入")
        self.clear_btn.clicked.connect(self.clear_input)
        button_row.addWidget(self.clear_btn)
        
        input_layout.addLayout(button_row)
        layout.addWidget(input_group)
        
        # 动作队列显示
        queue_group = QGroupBox("动作队列")
        queue_layout = QVBoxLayout(queue_group)
        
        self.queue_list = QListWidget()
        queue_layout.addWidget(self.queue_list)

        # 右侧不再重复“我的订单”
        
        queue_btn_layout = QHBoxLayout()
        self.clear_queue_btn = QPushButton("清空队列")
        self.clear_queue_btn.clicked.connect(self.clear_queue)
        queue_btn_layout.addWidget(self.clear_queue_btn)
        
        queue_layout.addLayout(queue_btn_layout)
        layout.addWidget(queue_group)
        
        # 日志显示
        log_group = QGroupBox("系统日志")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        return panel
        
    def setup_timer(self):
        """设置定时器更新状态"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_status)
        self.update_timer.start(1000)  # 每秒更新一次
        
        # 动作执行监控
        self.last_action_count = 0
        self.last_current_action = None
        
    def set_agents(self, agents: List[Any]):
        """设置agents列表"""
        self.agents = agents
        self.agent_combo.clear()
        for agent in self.agents:
            self.agent_combo.addItem(f"Agent {agent.agent_id}")
        if self.agents:
            self.target_agent = self.agents[0]
            # 延迟更新状态，确保UI完全初始化
            if hasattr(self, 'update_timer') and self.update_timer:
                self.update_timer.timeout.emit()  # 立即触发一次更新
                self.populate_poi_list()
            
    def on_agent_changed(self, agent_text):
        """Agent选择改变"""
        if not self.agents:
            return
        try:
            agent_id = agent_text.split()[-1]  # 提取Agent ID
            for agent in self.agents:
                if agent.agent_id == agent_id:
                    self.target_agent = agent
                    self.populate_poi_list()
                    self.log_message(f"🎯 已选择Agent {agent_id}")
                    self.log_message(f"   位置: ({agent.x:.1f}, {agent.y:.1f})")
                    self.log_message(f"   模式: {agent.mode.value}")
                    self.log_message(f"   控制模式: {'人类控制' if agent.human_control_mode else 'VLM控制'}")
                    break
        except Exception as e:
            self.log_message(f"选择Agent失败: {e}")
            
    def update_status(self):
        """更新状态显示"""
        if not self.target_agent:
            return
            
        # 检查UI组件是否已初始化
        if not hasattr(self, 'status_labels') or not self.status_labels:
            return
            
        try:
            status = self.target_agent.get_current_status()
            
            # 更新状态标签
            self.status_labels["position"].setText(f"({status['position'][0]:.1f}, {status['position'][1]:.1f})")
            self.status_labels["mode"].setText(status['mode'])
            self.status_labels["energy"].setText(f"{status['energy']:.1f}%")
            self.status_labels["earnings"].setText(f"${status['earnings']:.2f}")
            # 模拟时间（格式化为 mm:ss 或 hh:mm:ss）
            sim_s = float(status.get('sim_time_s', 0.0) or 0.0)
            h = int(sim_s // 3600); m = int((sim_s % 3600) // 60); s = int(sim_s % 60)
            sim_text = f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
            self.status_labels["sim_time"].setText(sim_text)
            self.status_labels["active_orders"].setText(str(len(status.get('active_orders', []) or [])))
            self.status_labels["carrying"].setText(str(len(status['carrying'])))
            self.status_labels["current_action"].setText(status['current_action'] or "无")
            
            # 更新控制模式显示
            if status['human_control_mode']:
                self.mode_label.setText("当前模式: 人类控制")
                self.mode_label.setStyleSheet("font-weight: bold; color: #32CD32;")
                self.toggle_mode_btn.setText("切换到VLM控制")
                self.submit_btn.setEnabled(True)
            else:
                self.mode_label.setText("当前模式: VLM控制")
                self.mode_label.setStyleSheet("font-weight: bold; color: #FF6347;")
                self.toggle_mode_btn.setText("切换到人类控制")
                self.submit_btn.setEnabled(False)
                
            # 更新动作队列显示
            self.update_queue_display()
            # 更新我的订单显示
            self.update_orders_display(status)
            
            # 监控动作执行状态
            self.monitor_action_execution(status)
            
        except Exception as e:
            self.log_message(f"更新状态时出错: {e}")
            
    def update_queue_display(self):
        """更新动作队列显示"""
        if not hasattr(self, 'queue_list') or not self.queue_list:
            return
            
        self.queue_list.clear()
        if self.target_agent and hasattr(self.target_agent, 'human_action_queue'):
            for i, action in enumerate(self.target_agent.human_action_queue):
                item_text = f"{i+1}. {action.kind.value}"
                if action.data:
                    item_text += f" - {action.data}"
                self.queue_list.addItem(item_text)

    def update_orders_display(self, status: Dict[str, Any]):
        """更新我的订单列表（活跃+帮助订单）"""
        if not hasattr(self, 'orders_list') or not self.orders_list:
            return
        try:
            self.orders_list.clear()
            # 优先使用 DeliveryMan 提供的轻量详情
            details = list(status.get('active_orders_detail') or [])
            if details:
                self.orders_list.addItem(f"活跃订单数: {len(details)}")
                for d in details:
                    oid = d.get('id')
                    pu  = d.get('pickup', '')
                    do  = d.get('dropoff', '')
                    picked = d.get('picked')
                    delivered = d.get('delivered')
                    line = f"#{oid} | {pu} -> {do} | picked={picked} delivered={delivered}"
                    self.orders_list.addItem(line)
            else:
                # 退化为仅显示 ID
                active_ids = list(status.get('active_orders') or [])
                self.orders_list.addItem(f"活跃订单数: {len(active_ids)}")
                if active_ids:
                    self.orders_list.addItem(f"IDs: {', '.join([str(i) for i in active_ids])}")
            # help orders if available
            if hasattr(self.target_agent, 'help_orders') and isinstance(self.target_agent.help_orders, dict):
                if self.target_agent.help_orders:
                    self.orders_list.addItem("—— 帮助订单 ——")
                for oid, o in (self.target_agent.help_orders or {}).items():
                    pu = getattr(o, 'pickup_road_name', '') or ''
                    do = getattr(o, 'dropoff_road_name', '') or ''
                    picked = getattr(o, 'has_picked_up', False)
                    delivered = getattr(o, 'has_delivered', False)
                    line = f"#{oid} | {pu} -> {do} | picked={picked} delivered={delivered}"
                    self.orders_list.addItem(line)
        except Exception as e:
            self.log_message(f"更新订单列表时出错: {e}")
                
    def monitor_action_execution(self, status):
        """监控动作执行状态"""
        if not self.target_agent:
            return
            
        try:
            current_action = status.get('current_action')
            queue_count = len(self.target_agent.human_action_queue) if hasattr(self.target_agent, 'human_action_queue') else 0
            
            # 检查是否有新动作开始执行
            if current_action and current_action != self.last_current_action:
                self.log_message(f"🚀 开始执行动作: {current_action}")
                self.last_current_action = current_action
                
            # 检查动作队列变化
            if queue_count != self.last_action_count:
                if queue_count < self.last_action_count:
                    self.log_message(f"✅ 动作已从队列中移除，剩余队列长度: {queue_count}")
                self.last_action_count = queue_count
                
            # 检查动作是否完成（从有动作变为无动作）
            if self.last_current_action and not current_action:
                self.log_message(f"✅ 动作执行完成: {self.last_current_action}")
                self.last_current_action = None
                
        except Exception as e:
            self.log_message(f"监控动作执行时出错: {e}")
                
    def toggle_control_mode(self):
        """切换控制模式"""
        if not self.target_agent:
            QMessageBox.warning(self, "警告", "没有选择目标Agent")
            return
            
        try:
            return
            current_mode = self.target_agent.human_control_mode
            self.target_agent.set_human_control_mode(not current_mode)
            new_mode = "人类" if not current_mode else "VLM"
            self.log_message(f"🔄 已切换到{new_mode}控制模式")
            if not current_mode:
                self.log_message(f"   现在可以提交人类动作了")
            else:
                self.log_message(f"   现在由VLM自动控制")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"切换控制模式失败: {e}")
            
    def on_action_selected(self, item):
        """动作列表项被选中"""
        action_type = item.data(256)
        self.action_combo.setCurrentText(action_type)
        self.on_action_type_changed(action_type)
        
    def on_action_type_changed(self, action_type):
        """动作类型改变时的处理"""
        # 根据动作类型设置默认参数提示
        default_params = {
            "move_to": '{"tx": 100.0, "ty": 200.0}',
            "rest": '{"target_pct": 100.0}',
            "wait": '{"duration_s": 30.0}',
            "charge_escooter": '{"target_pct": 100.0}',
            "buy": '{"item_id": "energy_drink", "qty": 1}',
            "say": '{"text": "Hello!"}',
            "turn_around": '{"angle": 180.0}',
        }
        
        if action_type in default_params:
            self.params_input.setText(default_params[action_type])
        else:
            self.params_input.setText("{}")
            
    def submit_action(self):
        """提交动作"""
        if not self.target_agent:
            QMessageBox.warning(self, "警告", "没有选择目标Agent")
            return
            
        if not self.target_agent.human_control_mode:
            QMessageBox.warning(self, "警告", "当前不是人类控制模式")
            return
            
        try:
            action_type = self.action_combo.currentText()
            params_text = self.params_input.text().strip()
            
            # 解析参数
            if params_text:
                params = json.loads(params_text)
            else:
                params = {}
                
            # 创建并提交动作
            action = self.target_agent.create_human_action(action_type, **params)
            success = self.target_agent.submit_human_action(action)
            
            if success:
                self.log_message(f"📤 已提交动作: {action_type}")
                if params:
                    self.log_message(f"   参数: {params}")
                self.log_message(f"   队列位置: {len(self.target_agent.human_action_queue)}")
                self.clear_input()
            else:
                self.log_message(f"❌ 提交动作失败: {action_type}")
                self.log_message(f"   可能原因: 不在人类控制模式或参数错误")
                
        except json.JSONDecodeError:
            QMessageBox.warning(self, "错误", "参数格式错误，请输入有效的JSON")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"提交动作失败: {e}")
            
    def clear_input(self):
        """清空输入"""
        self.params_input.clear()
        
    def clear_queue(self):
        """清空动作队列"""
        if self.target_agent and hasattr(self.target_agent, 'human_action_queue'):
            queue_count = len(self.target_agent.human_action_queue)
            self.target_agent.human_action_queue.clear()
            self.log_message(f"🗑️ 已清空动作队列 (清除了 {queue_count} 个动作)")
            
    def log_message(self, message):
        """添加日志消息"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        if hasattr(self, 'log_text') and self.log_text is not None:
            self.log_text.append(f"[{timestamp}] {message}")
        else:
            print(f"[{timestamp}] {message}")  # 备用输出

    def _poi_anchor_xy(self, meta: Dict[str, Any]) -> Optional[tuple]:
        """从 poi_meta 里选择一个可移动的锚点坐标（cm）。
        点状 POI（charging_station/bus_station）优先用 node.position（与 _nearest_poi_xy 一致）；
        建筑类优先 door -> dock -> node.position -> center。
        """
        try:
            node = meta.get("node")
            ptype = ""
            if node is not None:
                ptype = str(getattr(node, "type", "") or "").strip().lower()

            # 点状 POI：优先 node.position，保证与 _nearest_poi_xy 一致
            if ptype in ("charging_station", "bus_station"):
                if node is not None and hasattr(node, "position"):
                    return float(node.position.x), float(node.position.y)
                # 退化：dock -> center
                dock = meta.get("dock_node")
                if dock is not None and hasattr(dock, "position"):
                    return float(dock.position.x), float(dock.position.y)
                ctr = meta.get("center")
                if isinstance(ctr, (list, tuple)) and len(ctr) >= 2:
                    return float(ctr[0]), float(ctr[1])
                return None

            # 建筑类：door -> dock -> node.position -> center
            door = meta.get("door_node")
            if door is not None and hasattr(door, "position"):
                return float(door.position.x), float(door.position.y)

            dock = meta.get("dock_node")
            if dock is not None and hasattr(dock, "position"):
                return float(dock.position.x), float(dock.position.y)

            if node is not None and hasattr(node, "position"):
                return float(node.position.x), float(node.position.y)

            ctr = meta.get("center")
            if isinstance(ctr, (list, tuple)) and len(ctr) >= 2:
                return float(ctr[0]), float(ctr[1])
        except Exception:
            pass
        return None

    def populate_poi_list(self):
        """填充地图上所有可选 POI。"""
        self.actions_list.clear()
        if not self.target_agent or not hasattr(self.target_agent, "city_map"):
            return
        city_map = getattr(self.target_agent, "city_map", None)
        poi_meta_list = getattr(city_map, "poi_meta", None)
        if not isinstance(poi_meta_list, list):
            return

        for meta in poi_meta_list:
            # 名称与类型
            node = meta.get("node")
            name = ""
            ptype = ""
            try:
                name = (getattr(node, "display_name", "") or "").strip()
                if not name:
                    ptype = str(getattr(node, "type", "") or "").strip()
                    name = ptype or "POI"
                else:
                    ptype = str(getattr(node, "type", "") or "").strip()
            except Exception:
                pass

            # 道路名（可选）
            road_name = str(meta.get("road_name") or "")
            xy = self._poi_anchor_xy(meta)
            extra = f" @ {road_name}" if road_name else ""
            subtitle = f"{ptype}{extra}".strip()

            # 列表项显示
            title = name if name else "POI"
            item_text = title
            if subtitle:
                item_text += f"  ({subtitle})"

            it = QListWidgetItem(item_text)
            it.setData(256, {"name": title, "ptype": ptype, "road_name": road_name, "xy": xy})
            self.actions_list.addItem(it)

    def on_poi_selected(self, item: QListWidgetItem):
        """选择一个 POI 后，自动构造 move_to 动作参数。"""
        data = item.data(256) or {}
        xy = data.get("xy")
        if not xy:
            self.log_message("所选 POI 缺少有效坐标，无法生成移动目标")
            return
        tx, ty = float(xy[0]), float(xy[1])

        # 选中 move_to，并设置参数
        self.action_combo.setCurrentText("move_to")
        self.params_input.setText(json.dumps({"tx": tx, "ty": ty}))
        self.submit_btn.setEnabled(bool(self.target_agent and self.target_agent.human_control_mode))
        self.log_message(f"🎯 已选择 POI: {data.get('name') or ''} -> 目标({tx:.1f}, {ty:.1f})")