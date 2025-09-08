# -*- coding: utf-8 -*-
# Base/BusManager.py

import json
import math
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from Base.Bus import Bus, BusRoute, BusStop
from Base.Timer import VirtualClock

class BusManager:
    """公交系统管理器"""
    
    def __init__(self, clock: Optional[VirtualClock] = None):
        self.clock = clock if clock is not None else VirtualClock()
        self.routes: Dict[str, BusRoute] = {}
        self.buses: Dict[str, Bus] = {}
        self._update_timer = None
    
    def load_routes_from_world_data(self, world_data: Dict[str, Any]):
        """从世界数据加载公交路线"""
        bus_routes_data = world_data.get("bus_routes", [])
        
        for route_data in bus_routes_data:
            route_id = route_data.get("id", f"route_{len(self.routes)}")
            route_name = route_data.get("name", f"Route {route_id}")
            
            # 创建站点
            stops = []
            station_ids = route_data.get("station_ids", [])
            
            # 从世界节点中找到对应的站点
            world_nodes = world_data.get("nodes", [])
            node_map = {node.get("id"): node for node in world_nodes}
            
            for station_id in station_ids:
                if station_id in node_map:
                    node = node_map[station_id]
                    stop = BusStop(
                        id=station_id,
                        x=float(node.get("properties", {}).get("location", {}).get("x", 0)),
                        y=float(node.get("properties", {}).get("location", {}).get("y", 0)),
                        name=node.get("name", station_id),
                        wait_time_s=3.0  # 默认停靠10秒
                    )
                    stops.append(stop)
            
            # 获取路径点
            path_points = []
            path_data = route_data.get("path", [])
            
            # 构建交错路径：path[0], station0, path[1], station1, ...
            if path_data and stops:
                # 确定最大长度，以较长的为准
                max_length = max(len(path_data), len(stops))
                
                for i in range(max_length):
                    # 添加路径点（如果存在）
                    if i < len(path_data):
                        point = path_data[i]
                        if isinstance(point, dict):
                            x = float(point.get("x", 0)) * 100  # 转换为厘米
                            y = float(point.get("y", 0)) * 100  # 转换为厘米
                        elif isinstance(point, (list, tuple)) and len(point) >= 2:
                            x = float(point[0]) * 100  # 转换为厘米
                            y = float(point[1]) * 100  # 转换为厘米
                        else:
                            continue
                        path_points.append((x, y))
                    
                    # 添加站点（如果存在）
                    if i < len(stops):
                        stop = stops[i]
                        path_points.append((stop.x, stop.y))
            elif path_data:
                # 如果只有路径点没有站点，使用原始路径点
                for point in path_data:
                    if isinstance(point, dict):
                        x = float(point.get("x", 0)) * 100  # 转换为厘米
                        y = float(point.get("y", 0)) * 100  # 转换为厘米
                    elif isinstance(point, (list, tuple)) and len(point) >= 2:
                        x = float(point[0]) * 100  # 转换为厘米
                        y = float(point[1]) * 100  # 转换为厘米
                    else:
                        continue
                    path_points.append((x, y))
            elif stops:
                # 如果只有站点没有路径点，使用站点位置
                path_points = [(stop.x, stop.y) for stop in stops]
            
            # 创建路线
            route = BusRoute(
                id=route_id,
                name=route_name,
                stops=stops,
                path_points=path_points,
                speed_cm_s=1200.0,  # 默认速度
                direction=1
            )
            
            self.routes[route_id] = route
            print(f"Loaded bus route: {route_name} with {len(stops)} stops")
    
    def create_bus(self, bus_id: str, route_id: str) -> Optional[Bus]:
        """创建公交车"""
        if route_id not in self.routes:
            print(f"Route {route_id} not found")
            return None
        
        route = self.routes[route_id]
        bus = Bus(
            id=bus_id,
            route=route,
            clock=self.clock
        )
        
        self.buses[bus_id] = bus
        print(f"Created bus {bus_id} on route {route.name}")
        return bus
    
    def get_bus(self, bus_id: str) -> Optional[Bus]:
        """获取公交车"""
        return self.buses.get(bus_id)
    
    def get_route(self, route_id: str) -> Optional[BusRoute]:
        """获取公交路线"""
        return self.routes.get(route_id)
    
    def find_nearby_buses(self, x: float, y: float, radius_cm: float = 1000.0) -> List[Bus]:
        """查找附近的公交车"""
        nearby_buses = []
        for bus in self.buses.values():
            distance = math.hypot(bus.x - x, bus.y - y)
            if distance <= radius_cm:
                nearby_buses.append(bus)
        return nearby_buses
    
    def find_buses_at_stop(self, stop_id: str) -> List[Bus]:
        """查找在指定站点的公交车"""
        buses_at_stop = []
        for bus in self.buses.values():
            if bus.is_at_stop():
                current_stop = bus.get_current_stop()
                if current_stop and current_stop.id == stop_id:
                    buses_at_stop.append(bus)
        return buses_at_stop
    
    def find_nearest_bus_stop(self, x: float, y: float) -> Optional[Tuple[BusStop, float]]:
        """查找最近的公交站点"""
        nearest_stop = None
        nearest_distance = float('inf')
        
        for route in self.routes.values():
            for stop in route.stops:
                distance = math.hypot(stop.x - x, stop.y - y)
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_stop = stop
        
        if nearest_stop:
            return nearest_stop, nearest_distance
        return None
    
    def update_all_buses(self):
        """更新所有公交车状态"""
        for bus in self.buses.values():
            bus.update()
    
    def get_all_buses_status(self) -> List[str]:
        """获取所有公交车状态"""
        return [bus.get_status_text() for bus in self.buses.values()]
    
    def start_auto_update(self, interval_ms: int = 100):
        """启动自动更新（需要外部定时器调用）"""
        # 这里只是标记，实际更新需要外部调用 update_all_buses
        print(f"Bus system auto-update configured with {interval_ms}ms interval")
    
    def stop_auto_update(self):
        """停止自动更新"""
        print("Bus system auto-update stopped")
    
    def get_route_info(self, route_id: str) -> Optional[Dict[str, Any]]:
        """获取路线信息"""
        route = self.routes.get(route_id)
        if not route:
            return None
        
        return {
            "id": route.id,
            "name": route.name,
            "stops": [
                {
                    "id": stop.id,
                    "name": stop.name,
                    "x": stop.x,
                    "y": stop.y,
                    "wait_time_s": stop.wait_time_s
                }
                for stop in route.stops
            ],
            "path_points": route.path_points,
            "speed_cm_s": route.speed_cm_s,
            "direction": route.direction
        }
    
    def get_all_routes_info(self) -> Dict[str, Dict[str, Any]]:
        """获取所有路线信息"""
        return {
            route_id: self.get_route_info(route_id)
            for route_id in self.routes.keys()
        }
