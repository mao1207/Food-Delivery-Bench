# -*- coding: utf-8 -*-
# Scripts/test_bus_system.py

import sys
import os
import json

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

# 路径设置
SIMWORLD_DIR      = r"D:\Projects\Food-Delivery-Bench\SimWorld"
LLM_DELIVERY_DIR  = r"D:\Projects\Food-Delivery-Bench\LLM-Delivery"
sys.path.insert(0, SIMWORLD_DIR); sys.path.insert(0, LLM_DELIVERY_DIR)

from Base.Map import Map
from Base.Order import OrderManager
from Base.DeliveryMan import DeliveryMan, TransportMode, DMAction, DMActionKind
from Base.Store import StoreManager
from utils.map_observer import MapObserver
from Base.Timer import VirtualClock
from Base.Comms import init_comms

# 新增：公交系统
from Base.Bus import Bus, BusRoute, BusStop
from Base.BusManager import BusManager

ROADS_JSON        = r"D:\Projects\Food-Delivery-Bench\Test_Data\test\roads.json"
WORLD_JSON        = r"D:\Projects\Food-Delivery-Bench\Test_Data\test\progen_world_enriched.json"
STORE_ITEMS_JSON  = r"D:\Projects\Food-Delivery-Bench\LLM-Delivery\input\store_items.json"
FOOD_JSON         = r"D:\Projects\Food-Delivery-Bench\LLM-Delivery\input\food.json"
CONFIG_JSON       = r"D:\Projects\Food-Delivery-Bench\LLM-Delivery\input\config.json"
SPECIAL_NOTES_JSON = r"D:\Projects\Food-Delivery-Bench\LLM-Delivery\input\special_notes.json"

def _load_world_nodes(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("nodes", [])

def _load_cfg(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        return data
    except FileNotFoundError:
        raise RuntimeError(f"Config file not found: {path}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Config JSON parse error in {path}: {e}")

def main():
    app = QApplication(sys.argv)

    # --- 地图/订单/商店 ---
    m = Map(); m.import_roads(ROADS_JSON); m.import_pois(WORLD_JSON)
    nodes = _load_world_nodes(WORLD_JSON)

    # 读取 food.json
    with open(FOOD_JSON, "r", encoding="utf-8") as f:
        food_data = json.load(f) or {}
    menu_items = food_data.get("items", [])

    cfg = _load_cfg(CONFIG_JSON)

    # Clock
    clock = VirtualClock(time_scale=3.0)

    # Comms
    comms = init_comms(clock=clock)

    om = OrderManager(capacity=10, menu=menu_items, clock=clock)
    om.fill_pool(m, nodes)

    sm = StoreManager(); sm.load_items(STORE_ITEMS_JSON)

    # --- 公交系统初始化 ---
    bus_manager = BusManager(clock=clock)

    # 从世界数据加载公交路线
    with open(WORLD_JSON, "r", encoding="utf-8") as f:
        world_data = json.load(f)
    bus_manager.load_routes_from_world_data(world_data)

    # 创建公交车
    if bus_manager.routes:
        route_id = list(bus_manager.routes.keys())[0]  # 使用第一个路线
        bus1 = bus_manager.create_bus("bus_001", route_id)
        bus2 = bus_manager.create_bus("bus_002", route_id)
        print(f"Created buses on route {route_id}")
    else:
        print("No bus routes found in world data")

    # --- Viewer ---
    v = MapObserver(title="Bus System Test", clock=clock)
    v.draw_map(m, WORLD_JSON, show_bus=True, show_docks=False,
               show_building_links=True, show_road_names=True, plain_mode="pudo")
    v.resize(1200, 900); v.show()
    v.attach_order_manager(om)
    v.attach_comms(comms)
    v.attach_bus_manager(bus_manager)  # 绑定公交管理器

    # --- 工具：随机道路坐标 ---
    def rand_xy():
        xy = v.random_xy_on_roads()
        return xy if xy else (0.0, 0.0)

    # --- 初始化一个测试 agent ---
    ax, ay = rand_xy()
    dm = DeliveryMan("test_agent", m, nodes, ax, ay, mode=TransportMode.WALK, clock=clock)

    dm.bind_viewer(v)
    dm.set_order_manager(om)
    dm.set_store_manager(sm)
    dm.set_bus_manager(bus_manager)  # 绑定公交管理器
    dm.vlm_add_memory("prefers_using_public_transport")
    dm.register_to_comms()

    # --- 演示公交使用 ---
    def demo_bus_usage():
        print("\n=== 公交系统演示 ===")

        # 1. 查找最近的公交站点
        nearest_stop, distance = bus_manager.find_nearest_bus_stop(dm.x, dm.y)
        if nearest_stop:
            print(f"Nearest bus stop: {nearest_stop.name or nearest_stop.id} at distance {distance/100:.1f}m")

            # 2. 移动到公交站点
            print(f"Moving to bus stop...")
            dm.enqueue_action(DMAction(DMActionKind.MOVE_TO, data={
                "tx": nearest_stop.x, 
                "ty": nearest_stop.y, 
                "use_route": True, 
                "snap_cm": 120.0
            }))

            # 3. 等车
            def wait_for_bus():
                print("Waiting for bus...")
                dm.enqueue_action(DMAction(DMActionKind.WAIT_FOR_BUS, data={
                    "stop_id": nearest_stop.id,
                    "max_wait_s": 60.0
                }))

            # 4. 上车（当到达站点后）
            def board_bus():
                if bus_manager.buses:
                    bus_id = list(bus_manager.buses.keys())[0]
                    print(f"Boarding bus {bus_id}...")
                    dm.enqueue_action(DMAction(DMActionKind.BOARD_BUS, data={
                        "bus_id": bus_id,
                        "target_stop": ""  # 可以指定目标站点
                    }))

            # 延迟执行等车和上车
            QTimer.singleShot(3000, wait_for_bus)  # 3秒后等车
            QTimer.singleShot(8000, board_bus)     # 8秒后上车
        else:
            print("No bus stops found!")

    # 启动演示
    QTimer.singleShot(2000, demo_bus_usage)

    # --- 状态打印定时器 ---
    def print_status():
        print("\n=== 系统状态 ===")
        print(f"Agent: {dm.to_text()}")
        print("Buses:")
        for status in bus_manager.get_all_buses_status():
            print(f"  {status}")
        print("-" * 60)

    status_timer = QTimer(v)
    status_timer.setInterval(10000)  # 每10秒打印一次
    status_timer.timeout.connect(print_status)
    status_timer.start()

    # 启动agent
    dm.kickstart()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()