from Base.Types import Vector
from Config.config import Config

import matplotlib.pyplot as plt
from Base.Map import Map
from typing import Optional


def estimated_delivery_time(store_position: Vector, customer_position: Vector):
    distance = (store_position - customer_position).length()
    return (distance / Config.DELIVERY_MAN_MIN_SPEED) * 3

def visualize_map(map_obj: Map, save_path: Optional[str] = None):
    """
    可视化地图对象，显示所有节点和边
    
    参数:
        map_obj: Map对象
        save_path: 可选，保存图片的路径。如果为None则显示图片
    """
    # 创建新的图形
    plt.figure(figsize=(12, 8))
    
    # 绘制边
    for edge in map_obj.edges:
        x_coords = [edge.node1.position.x, edge.node2.position.x]
        y_coords = [edge.node1.position.y, edge.node2.position.y]
        plt.plot(x_coords, y_coords, 'gray', alpha=0.5, linewidth=1)
    
    # 收集不同类型的节点
    normal_nodes = []
    intersection_nodes = []
    supply_nodes = []
    
    for node in map_obj.nodes:
        if node.type == "normal":
            normal_nodes.append(node)
        elif node.type == "intersection":
            intersection_nodes.append(node)
        elif node.type == "supply":
            supply_nodes.append(node)
    
    # 绘制不同类型的节点
    if normal_nodes:
        x_coords = [node.position.x for node in normal_nodes]
        y_coords = [node.position.y for node in normal_nodes]
        plt.scatter(x_coords, y_coords, c='blue', s=30, label='Normal Nodes')
    
    if intersection_nodes:
        x_coords = [node.position.x for node in intersection_nodes]
        y_coords = [node.position.y for node in intersection_nodes]
        plt.scatter(x_coords, y_coords, c='red', s=50, label='Intersections')
    
    if supply_nodes:
        x_coords = [node.position.x for node in supply_nodes]
        y_coords = [node.position.y for node in supply_nodes]
        plt.scatter(x_coords, y_coords, c='green', s=80, marker='*', label='Supply Points')
    
    # 设置图形属性
    plt.title('Map Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保持横纵比例相等
    plt.axis('equal')
    
    # 保存或显示图形
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()