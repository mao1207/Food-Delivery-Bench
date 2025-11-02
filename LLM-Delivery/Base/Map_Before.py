import random
from collections import defaultdict
from typing import Optional, List
from Base.Types import Vector, Node, Edge, Road
from Config import Config
from collections import deque
import os
import json

class Map:
    def __init__(self):
        self.nodes = set()
        self.edges = set()
        self.adjacency_list = defaultdict(list)

    def __str__(self):
        return f"Nodes: {self.nodes}\nEdges: {self.edges}\n"

    def __repr__(self):
        return self.__str__()

    def add_node(self, node: Node):
        self.nodes.add(node)

    def add_edge(self, edge: Edge):
        self.edges.add(edge)
        self.adjacency_list[edge.node1].append(edge.node2)
        self.adjacency_list[edge.node2].append(edge.node1)

    def get_adjacency_list(self):
        return self.adjacency_list

    def get_adjacent_points(self, node: Node):
        points = [n.position for n in self.adjacency_list[node]]
        return points

    def has_edge(self, edge: Edge):
        return edge in self.edges

    def get_points(self):
        return [node.position for node in self.nodes]

    def get_nodes(self):
        return self.nodes

    def get_random_node(self, exclude_pos: Optional[List[Node]] = None):
        nodes = [node for node in self.nodes if node.type == "normal"]
        if exclude_pos:
            nodes = [node for node in nodes if node not in exclude_pos]

        if not exclude_pos or len(exclude_pos) == 0:
            return random.choice(nodes) 

        max_min_dist = -1
        best_node = None
        for candidate in nodes:
            min_dist = min(candidate.position.distance(p.position) for p in exclude_pos)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_node = candidate

        return best_node if best_node else random.choice(nodes)


    def get_random_node_with_distance(self, base_pos: List[Node], exclude_pos: Optional[List[Node]] = None, min_distance: float = 0, max_distance: float = 100000):
        nodes = [node for node in self.nodes if node.type != "intersection" and node.type != "supply"]
        if exclude_pos:
            nodes = [node for node in nodes if node not in exclude_pos]

        candidates = []
        for node in nodes:
            for base in base_pos:
                d = node.position.distance(base.position)
                if min_distance <= d <= max_distance:
                    candidates.append((node, d))
                    break

        if not candidates:
            return random.choice(nodes)

        best_node = max(candidates, key=lambda x: min(x[0].position.distance(p.position) for p in base_pos))[0]
        return best_node


    def get_random_node_with_edge_distance(self, base_pos: List[Node], exclude_pos: Optional[List[Node]] = None, min_distance: float = 0, max_distance: float = 200):
        # get a random node that is at least min_distance away from any nodes in exclude_pos
        nodes = [node for node in self.nodes if node.type != "intersection" and node.type != "supply"]
        if exclude_pos:
            nodes = [node for node in nodes if node not in exclude_pos]
        base_node = random.choice(base_pos)
        target_distance = random.randint(min_distance, max_distance)

        # 使用BFS找到最接近目标距离的节点
        queue = deque([(base_node, 0)])  # (node, distance) pairs
        visited = {base_node}
        best_distance_diff = float('inf')
        result_nodes = []

        while queue:
            current_node, current_distance = queue.popleft()
            # 计算与目标距离的差值
            distance_diff = abs(current_distance - target_distance)
            # 如果找到更接近的节点，更新结果
            if distance_diff < best_distance_diff:
                best_distance_diff = distance_diff
                if (not exclude_pos or current_node not in exclude_pos) and current_node.type != "intersection" and current_node.type != "supply":
                    result_nodes = [current_node]
            # 如果找到相同距离差的节点，添加到结果中
            elif distance_diff == best_distance_diff:
                if (not exclude_pos or current_node not in exclude_pos) and current_node.type != "intersection" and current_node.type != "supply":
                    result_nodes.append(current_node)

            # 继续探索邻居节点
            for neighbor in self.adjacency_list[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, current_distance + 1))

        if not result_nodes:
            result_nodes.append(random.choice(nodes))

        return result_nodes  # 返回所有最接近目标距离的节点

    def get_supply_points(self):
        return [node.position for node in self.nodes if node.type == "supply"]

    def connect_adjacent_roads(self):
        """
        Connect nodes from adjacent roads that are close to each other
        """
        nodes = list(self.nodes)
        connection_threshold = Config.SIDEWALK_OFFSET * 2 + 100   # Reasonable threshold for connecting nearby nodes


        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                node1 = nodes[i]
                node2 = nodes[j]

                # If nodes are close enough and not already connected
                if (node1.position.distance(node2.position) < connection_threshold and
                    not self.has_edge(Edge(node1, node2))):
                    self.add_edge(Edge(node1, node2))

    def interpolate_nodes(self):
        """
        Interpolate nodes between existing nodes to create a smoother map
        """
        current_edges = list(self.edges)

        for edge in current_edges:
            distance = edge.weight
            num_points = int(distance / (2 * Config.SIDEWALK_OFFSET))

            if num_points <= 1:
                continue

            direction = (edge.node2.position - edge.node1.position).normalize()

            new_nodes = []

            supply_point_index = random.randint(2, num_points - 2) if num_points > 1 else None

            for i in range(1, num_points + 1):
                new_point = edge.node1.position + direction * (i * 2 * Config.SIDEWALK_OFFSET)
                node_type = "supply" if i == supply_point_index else "normal"
                new_node = Node(new_point, type=node_type)
                self.add_node(new_node)
                new_nodes.append(new_node)

            self.edges.remove(edge)
            self.adjacency_list[edge.node1].remove(edge.node2)
            self.adjacency_list[edge.node2].remove(edge.node1)

            all_nodes = [edge.node1] + new_nodes + [edge.node2]
            for i in range(len(all_nodes) - 1):
                self.add_edge(Edge(all_nodes[i], all_nodes[i + 1]))

    def get_edge_distance_between_two_points(self, point1: Node, point2: Node) -> int:
        """Calculate the minimum edge distance between two points using BFS.
        Args:
            point1: Starting node
            point2: Target node

        Returns:
            The minimum number of edges between the two points
        """
        if point1 == point2:
            return 0

        queue = deque([(point1, 0)])  # (node, distance) pairs
        visited = {point1}

        while queue:
            current_point, distance = queue.popleft()

            # Check if we've reached the target
            if current_point == point2:
                return distance

            # Explore neighbors
            for neighbor in self.adjacency_list[current_point]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        # If we get here, no path was found
        raise ValueError(f"No path found between {point1} and {point2}")

    def import_map(self, map_path: str):
        """
        Import a map from a JSON file
        """
        with open(map_path, 'r') as f:
            roads_data = json.load(f)

        roads = roads_data['roads']

        road_objects = []
        for road in roads:
            start = Vector(road['start']['x']*100, road['start']['y']*100)
            end = Vector(road['end']['x']*100, road['end']['y']*100)
            road_objects.append(Road(start, end))

        # Initialize the map
        for road in road_objects:
            normal_vector = Vector(road.direction.y, -road.direction.x)
            point1 = road.start - normal_vector * (Config.SIDEWALK_OFFSET) + road.direction * Config.SIDEWALK_OFFSET
            point2 = road.end - normal_vector * (Config.SIDEWALK_OFFSET) - road.direction * Config.SIDEWALK_OFFSET

            point3 = road.end + normal_vector * (Config.SIDEWALK_OFFSET) - road.direction * Config.SIDEWALK_OFFSET
            point4 = road.start + normal_vector * (Config.SIDEWALK_OFFSET) + road.direction * Config.SIDEWALK_OFFSET

            node1 = Node(point1, "intersection")
            node2 = Node(point2, "intersection")
            node3 = Node(point3, "intersection")
            node4 = Node(point4, "intersection")

            self.add_node(node1)
            self.add_node(node2)
            self.add_node(node3)
            self.add_node(node4)

            self.add_edge(Edge(node1, node2))
            self.add_edge(Edge(node3, node4))
            self.add_edge(Edge(node1, node4))
            self.add_edge(Edge(node2, node3))
        # Connect adjacent roads by finding nearby nodes
        self.connect_adjacent_roads()
        self.interpolate_nodes()
