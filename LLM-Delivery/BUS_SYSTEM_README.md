# 公交系统使用说明

## 概述

实现了完整的公交系统，包括：
- 公交车沿路线自动行驶
- 站点停靠和等待时间
- 乘客上下车功能
- 可视化显示

## 核心组件

### 1. Bus类 (`Base/Bus.py`)
- 管理单辆公交车的状态、位置、速度
- 支持停靠、行驶、等待状态
- 乘客管理（上车/下车）

### 2. BusManager类 (`Base/BusManager.py`)
- 管理所有公交路线和车辆
- 从世界数据加载公交路线
- 提供查找附近公交、站点等功能

### 3. DeliveryMan扩展
- 新增公交相关动作：`BOARD_BUS`, `ALIGHT_BUS`, `WAIT_FOR_BUS`
- 支持乘坐公交模式：`TransportMode.BUS`
- 自动跟随公交位置移动

### 4. MapObserver扩展
- 显示公交车位置（橙色方块）
- 显示公交状态文字（当前站点/下一站点/乘客数）

## 使用方法

### 1. 初始化公交系统

```python
from Base.BusManager import BusManager

# 创建公交管理器
bus_manager = BusManager(clock=clock)

# 从世界数据加载路线
with open("world.json", "r") as f:
    world_data = json.load(f)
bus_manager.load_routes_from_world_data(world_data)

# 创建公交车
bus = bus_manager.create_bus("bus_001", "route_bus_1")
```

### 2. 绑定到DeliveryMan

```python
# 绑定公交管理器
dm.set_bus_manager(bus_manager)

# 绑定到viewer
v.attach_bus_manager(bus_manager)
```

### 3. 使用公交动作

```python
from Base.DeliveryMan import DMAction, DMActionKind

# 等车
dm.enqueue_action(DMAction(DMActionKind.WAIT_FOR_BUS, data={
    "stop_id": "bus_station_1",
    "max_wait_s": 300.0
}))

# 上车
dm.enqueue_action(DMAction(DMActionKind.BOARD_BUS, data={
    "bus_id": "bus_001",
    "target_stop": "bus_station_5"  # 可选：目标站点
}))

# 下车
dm.enqueue_action(DMAction(DMActionKind.ALIGHT_BUS, data={
    "stop_id": "bus_station_5"  # 可选：指定站点
}))
```

## 测试

运行测试脚本：
```bash
python Scripts/test_bus_system.py
```

该脚本会：
1. 加载世界数据和公交路线
2. 创建公交车并开始运行
3. 创建一个测试agent
4. 演示移动到公交站点、等车、上车的完整流程

## 特性

- **自动行驶**：公交车沿路线自动移动，在站点停靠
- **乘客管理**：支持多乘客同时乘坐
- **状态同步**：乘客位置自动跟随公交车
- **可视化**：实时显示公交位置和状态
- **灵活配置**：可配置停靠时间、行驶速度等

## 注意事项

1. 需要世界数据包含 `bus_routes` 和 `bus_station` 节点
2. 公交路线需要包含 `path` 路径点数据
3. 乘客必须在公交附近（5米内）才能上车
4. 公交必须在站点停靠时才能上下车
