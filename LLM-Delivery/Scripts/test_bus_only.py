# -*- coding: utf-8 -*-
# Scripts/test_bus_only.py - 只显示公交移动，不需要deliveryman

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
from utils.map_observer import MapObserver
from Base.Timer import VirtualClock

# 公交系统
from Base.Bus import Bus, BusRoute, BusStop
from Base.BusManager import BusManager

ROADS_JSON        = r"D:\Projects\Food-Delivery-Bench\Test_Data\test\roads.json"
WORLD_JSON        = r"D:\Projects\Food-Delivery-Bench\Test_Data\test\progen_world_enriched.json"

def _load_world_nodes(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("nodes", [])

def main():
    app = QApplication(sys.argv)

    # --- 地图初始化 ---
    m = Map(); m.import_roads(ROADS_JSON); m.import_pois(WORLD_JSON)
    nodes = _load_world_nodes(WORLD_JSON)

    # Clock
    clock = VirtualClock(time_scale=1.0)  # 3倍时间加速

    # --- 公交系统初始化 ---
    bus_manager = BusManager(clock=clock)
    
    # 从世界数据加载公交路线
    with open(WORLD_JSON, "r", encoding="utf-8") as f:
        world_data = json.load(f)
    bus_manager.load_routes_from_world_data(world_data)
    
    # 创建多辆公交车
    if bus_manager.routes:
        route_id = list(bus_manager.routes.keys())[0]  # 使用第一个路线
        print(f"Creating buses on route: {route_id}")
        
        # 创建3辆公交车，错开发车时间
        bus1 = bus_manager.create_bus("bus_001", route_id)
        bus2 = bus_manager.create_bus("bus_002", route_id)
        # bus3 = bus_manager.create_bus("bus_003", route_id)
        
        # # 让第二辆和第三辆车错开发车
        if bus2 and bus2.route.stops:
            bus2.current_path_index = 2
            bus2.current_stop_index = 0  # 从第二个站点开始
            bus2.x = bus2.route.stops[1].x
            bus2.y = bus2.route.stops[1].y
            bus2.state = bus2.state.__class__.MOVING
            
        # if bus3 and bus3.route.stops:
        #     bus3.current_stop_index = 2  # 从第三个站点开始
        #     bus3.x = bus3.route.stops[2].x
        #     bus3.y = bus3.route.stops[2].y
        #     bus3.state = bus3.state.__class__.STOPPED
        #     bus3.stop_start_time = clock.now_sim() - 3.0  # 提前3秒到达
            
        print(f"Created {len(bus_manager.buses)} buses")
    else:
        print("No bus routes found in world data")

    # --- Viewer ---
    v = MapObserver(title="Bus Movement Only", clock=clock)
    v.draw_map(m, WORLD_JSON, show_bus=True, show_docks=False,
               show_building_links=True, show_road_names=True, plain_mode="pudo")
    v.resize(1200, 900); v.show()
    v.attach_bus_manager(bus_manager)  # 绑定公交管理器

    # --- 状态打印定时器 ---
    def print_status():
        print("\n=== 公交系统状态 ===")
        print("Buses:")
        for status in bus_manager.get_all_buses_status():
            print(f"  {status}")
        print("-" * 60)

    status_timer = QTimer(v)
    status_timer.setInterval(1000)  # 每5秒打印一次
    status_timer.timeout.connect(print_status)
    status_timer.start()

    # 启动时打印一次状态
    QTimer.singleShot(1000, print_status)

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
