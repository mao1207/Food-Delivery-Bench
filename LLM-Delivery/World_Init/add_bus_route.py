# enrich_world_roads_plus_bbox.py
# -*- coding: utf-8 -*-
import json, math, random
from collections import defaultdict
from pathlib import Path

# ========= IO =========
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ========= Geometry =========
def dist(p, q):
    return math.hypot(p["x"] - q["x"], p["y"] - q["y"])

def cumulative_lengths(points):
    acc = [0.0]
    for i in range(1, len(points)):
        acc.append(acc[-1] + dist(points[i - 1], points[i]))
    return acc

def heading_deg(p, q):
    return math.degrees(math.atan2(q["y"] - p["y"], q["x"] - p["x"]))

def unit_tangent(p, q):
    dx, dy = (q["x"] - p["x"]), (q["y"] - p["y"])
    L = math.hypot(dx, dy)
    if L == 0:
        return 1.0, 0.0
    return dx / L, dy / L

def left_normal(p, q):
    tx, ty = unit_tangent(p, q)
    return -ty, tx  # 左法向

# ========= World helpers =========
def collect_buildings(world):
    return [n for n in world.get("nodes", [])
            if n.get("instance_name", "").startswith("BP_Building")]

def normalize_roads_list(roads):
    return roads.get("roads") if isinstance(roads, dict) and "roads" in roads else roads

def build_road_graph_from_roads(roads_json):
    graph = defaultdict(list)
    node_map = {}
    seen = set()
    seg_cnt = 0

    def key(p, nd=4):
        return (round(p["x"], nd), round(p["y"], nd))

    roads_list = normalize_roads_list(roads_json) or []
    for seg in roads_list:
        s, e = seg.get("start"), seg.get("end")
        if not (s and e):
            continue
        u, v = key(s), key(e)
        if u == v:
            continue
        if u not in seen:
            node_map[u] = {"x": u[0], "y": u[1]}
            seen.add(u)
        if v not in seen:
            node_map[v] = {"x": v[0], "y": v[1]}
            seen.add(v)
        w = dist(node_map[u], node_map[v])
        if w == 0:
            continue
        graph[u].append((v, w))
        graph[v].append((u, w))
        seg_cnt += 1

    print(f"[build_road_graph_from_roads] segments={seg_cnt}, nodes={len(node_map)}")
    return graph, node_map

def extract_segments_from_roads(roads_json):
    segments = []
    roads_list = normalize_roads_list(roads_json) or []
    for seg in roads_list:
        s, e = seg.get("start"), seg.get("end")
        if not (s and e):
            continue
        p = {"x": float(s.get("x", 0.0)), "y": float(s.get("y", 0.0))}
        q = {"x": float(e.get("x", 0.0)), "y": float(e.get("y", 0.0))}
        if p["x"] == q["x"] and p["y"] == q["y"]:
            continue
        segments.append((p, q))
    return segments

# ========= ID 生成（非 UUID，类似 GEN_xxx_序号_随机数） =========
class IdFactory:
    def __init__(self, existing_ids=None):
        self.counters = defaultdict(int)
        self.existing = set(existing_ids or [])

    def new_id(self, instance_name):
        self.counters[instance_name] += 1
        seq = self.counters[instance_name] - 1
        for _ in range(100):
            rid = random.randint(0, 999)
            nid = f"GEN_{instance_name}_{seq}_{rid}"
            if nid not in self.existing:
                self.existing.add(nid)
                return nid
        # 最差兜底（基本不会走到）
        nid = f"GEN_{instance_name}_{seq}_{random.randint(1000,9999)}"
        self.existing.add(nid)
        return nid

# ========= 点状 POI 写入 =========
def add_point_poi(world, id_factory, instance_name, poi_type, x, y, yaw=0.0):
    nid = id_factory.new_id(instance_name)
    node = {
        "id": nid,
        "instance_name": instance_name,
        "properties": {
            "poi_type": poi_type,
            "location": {"x": float(x), "y": float(y), "z": 0.0},
            "orientation": {"pitch": 0.0, "yaw": float(yaw), "roll": 0.0}
        }
    }
    world.setdefault("nodes", []).append(node)
    return nid

# ========= 公交路线 & 站点（带最小间距） =========
def generate_route_polyline(graph, node_map, target_len=30000.0, max_steps=6000):
    if not graph:
        return []
    nodes = list(graph.keys())
    cur = random.choice(nodes)
    route_nodes = [cur]
    total = 0.0
    visited_edges = set()
    steps = 0
    while total < target_len and steps < max_steps:
        steps += 1
        nbrs = graph[cur]
        if not nbrs:
            break
        unvisited, visited = [], []
        for v, w in nbrs:
            ek = (cur, v) if cur < v else (v, cur)
            (unvisited if ek not in visited_edges else visited).append((v, w, ek))
        cand = unvisited if unvisited else visited
        if not cand:
            break
        v, w, ek = random.choice(cand)
        visited_edges.add(ek)
        route_nodes.append(v)
        total += w
        cur = v
    return [node_map[n] for n in route_nodes]

def place_bus_stops_along(polyline, num_stops=6, min_gap_m=50.0,
                          spacing_jitter_frac=0.2, lateral_offset_m=3.0, scale=100.0):
    """
    - 均匀取点：以 total_len/(num_stops+1) 为基距，再加抖动（±spacing_jitter_frac）
    - 保障最小间距 min_gap_m
    - 每个点随机左右偏移 lateral_offset_m，yaw 正对中心线
    - 返回 [{'x','y','yaw'}]，x/y 已 ×scale
    """
    if len(polyline) < 2 or num_stops <= 0:
        return []
    cum = cumulative_lengths(polyline)
    total = cum[-1]
    if total <= 0:
        return []

    base_step = total / (num_stops + 1)
    targets = []
    last_pos = 0.0
    for i in range(1, num_stops + 1):
        # 基于等间距 + 抖动
        center = base_step * i
        jitter = (random.uniform(-spacing_jitter_frac, spacing_jitter_frac)) * base_step
        pos = max(0.0, min(total, center + jitter))
        # 与上一个目标保持最小间距
        if targets and pos - targets[-1] < min_gap_m:
            pos = targets[-1] + min_gap_m
            pos = min(pos, total)
        if i < num_stops and (total - pos) < (num_stops - i) * min_gap_m:
            # 保证后续还能放得下
            pos = max(pos - (min_gap_m / 2.0), 0.0)
        targets.append(pos)

    # 投影到折线段上
    stops, seg_idx = [], 0
    for tlen in targets:
        while seg_idx + 1 < len(cum) and cum[seg_idx + 1] < tlen:
            seg_idx += 1
        p = polyline[seg_idx]
        q = polyline[seg_idx + 1] if seg_idx + 1 < len(polyline) else p
        L = dist(p, q)
        if L == 0:
            stops.append({"x": p["x"] * scale, "y": p["y"] * scale, "yaw": 0.0})
            continue

        alpha = (tlen - cum[seg_idx]) / L
        alpha = max(0.0, min(1.0, alpha))

        # 中心线点
        cx = p["x"] + (q["x"] - p["x"]) * alpha
        cy = p["y"] + (q["y"] - p["y"]) * alpha

        # 随机左右偏移
        side_sign = random.choice((+1, -1))
        nx, ny = left_normal(p, q)
        px = cx + nx * (lateral_offset_m * side_sign)
        py = cy + ny * (lateral_offset_m * side_sign)

        # 朝向正对中心线
        yaw = heading_deg(p, q) - side_sign * 180.0
        stops.append({"x": px * scale, "y": py * scale, "yaw": yaw})
    return stops

# ========= 沿路放置：充电桩 =========
def add_chargers_along_roads(world, id_factory, roads_json, count=10, offset_m=2.0, scale=100.0):
    segments = extract_segments_from_roads(roads_json)
    if not segments or count <= 0:
        return 0

    placed = 0
    for i in range(count):
        p, q = random.choice(segments)
        # 在线段上取一点
        t = random.uniform(0.15, 0.85)
        cx = p["x"] + (q["x"] - p["x"]) * t
        cy = p["y"] + (q["y"] - p["y"]) * t

        # 随机左右
        side_sign = random.choice((+1, -1))
        nx, ny = left_normal(p, q)
        px = cx + nx * (offset_m * side_sign)
        py = cy + ny * (offset_m * side_sign)

        # 正对中心线
        yaw = heading_deg(p, q)

        add_point_poi(
            world, id_factory,
            instance_name=f"POI_ChargingStation_Road",
            poi_type="charging_station",
            x=px * scale, y=py * scale, yaw=yaw
        )
        placed += 1
    return placed

# ========= BBox 合并（一步到位内置） =========
def merge_bbox_into_world_inplace(world_obj, bbox_obj,
                                  key_in_world="instance_name",
                                  bbox_root_key="buildings",
                                  write_to="properties.bbox",
                                  default_zero_for_buildings=True,
                                  also_copy_keys=None):
    """
    将 bbox.json 的信息合并到 world 的节点中（按 instance_name 匹配）。
    - 对于建筑（instance_name 以 BP_Building 开头）且找不到 bbox 的，写默认 0/0/0（可关）
    - 其它类型节点（POI_xxx）不强制写 bbox
    """
    if also_copy_keys is None:
        also_copy_keys = []

    buildings_map = bbox_obj.get(bbox_root_key, {}) or {}
    items = world_obj.get("nodes", []) if isinstance(world_obj, dict) else world_obj

    def set_nested(holder: dict, dotted_key: str, value):
        cur = holder
        *parents, last = dotted_key.split(".")
        for k in parents:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        cur[last] = value

    hit = miss = 0
    for obj in items:
        name = obj.get(key_in_world)
        if not name:
            miss += 1
            continue

        info = buildings_map.get(name)
        if info and "bbox" in info:
            set_nested(obj, write_to, info["bbox"])
            for k in also_copy_keys:
                if k in info:
                    set_nested(obj, f"properties.{k}", info[k])
            hit += 1
        else:
            # 仅对建筑缺省写 0/0/0
            if default_zero_for_buildings and isinstance(name, str) and name.startswith("BP_Building"):
                set_nested(obj, write_to, {"x": 0, "y": 0, "z": 0})
            miss += 1
    print(f"[merge_bbox] matched={hit}, others={miss}")

# ========= 建筑打标签 =========
def tag_random_buildings(world, k, label, exclude_ids):
    buildings = collect_buildings(world)
    cand = [b for b in buildings
            if b.get("id") not in exclude_ids and b.get("properties", {}).get("poi_type") is None]
    if not cand:
        return []
    chosen = random.sample(cand, k=min(k, len(cand)))
    for b in chosen:
        b.setdefault("properties", {})["poi_type"] = label
    exclude_ids.update(b["id"] for b in chosen)
    return chosen

# ========= 主入口 =========
def enrich_world_in_place(
    world_json_path,
    output_path,
    roads_json_path=None,
    bbox_json_path=None,
    # 建筑标签
    num_restaurants=5,
    num_stores=5,
    num_rest_areas=2,
    num_hospitals=1,
    num_car_rentals=2,
    # 公交参数
    bus_route_target_len=30000.0,
    num_bus_stops=10,
    bus_min_gap_m=50.0,
    bus_spacing_jitter_frac=0.2,
    bus_offset_m=3.0,
    # 充电桩
    road_chargers=10,
    charger_offset_m=2.0,
    # 缩放
    scale=100.0,     # 坐标整体 ×100（米->厘米）
    rng_seed=None
):
    if rng_seed is not None:
        random.seed(rng_seed)
    else:
        random.seed()

    world = load_json(world_json_path)

    # 现有 id 收集，用于避免冲突
    existing_ids = {n.get("id") for n in world.get("nodes", []) if isinstance(n, dict) and "id" in n}
    id_factory = IdFactory(existing_ids=existing_ids)

    # 1) 建筑打标签（包含你提到的 rest_area / hospital / car_rental）
    exclude_ids = set()
    restaurants   = tag_random_buildings(world, num_restaurants, "restaurant", exclude_ids)
    stores        = tag_random_buildings(world, num_stores, "store", exclude_ids)
    rest_areas    = tag_random_buildings(world, num_rest_areas, "rest_area", exclude_ids)
    hospitals     = tag_random_buildings(world, num_hospitals, "hospital", exclude_ids)
    car_rentals   = tag_random_buildings(world, num_car_rentals, "car_rental", exclude_ids)

    # 2) 道路数据（公交 & 沿路充电桩）
    bus_routes = []
    graph = node_map = None
    roads_json = None
    if roads_json_path:
        roads_json = load_json(roads_json_path)
        graph, node_map = build_road_graph_from_roads(roads_json)

        # 2a) 沿路放充电桩
        added_road_chargers = add_chargers_along_roads(
            world, id_factory, roads_json,
            count=road_chargers, offset_m=charger_offset_m, scale=scale
        )
    else:
        added_road_chargers = 0

    # 3) 生成公交路线 + 站点（带最小间距）
    if graph:
        polyline = generate_route_polyline(graph, node_map,
                                           target_len=bus_route_target_len,
                                           max_steps=6000)
        print(f"Generated bus route polyline with {len(polyline)} points.")
        if len(polyline) >= 2:
            stops = place_bus_stops_along(polyline,
                                          num_stops=num_bus_stops,
                                          min_gap_m=bus_min_gap_m,
                                          spacing_jitter_frac=bus_spacing_jitter_frac,
                                          lateral_offset_m=bus_offset_m,
                                          scale=scale)
            station_ids = []
            for i, pt in enumerate(stops):
                nid = add_point_poi(world, id_factory,
                                    instance_name="POI_BusStation",
                                    poi_type="bus_station",
                                    x=pt["x"], y=pt["y"], yaw=pt["yaw"])
                station_ids.append(nid)
            bus_routes.append({
                "type": "bus_route",
                "id": "route_bus_1",
                "station_ids": station_ids,
                "path": polyline  # 路径保持原单位（未×100）
            })
            world["bus_routes"] = bus_routes
    else:
        print("No roads graph provided for bus route; skip bus.")

    # 4) BBox 合并（一步到位）
    if bbox_json_path:
        bbox_obj = load_json(bbox_json_path)
        merge_bbox_into_world_inplace(
            world_obj=world,
            bbox_obj=bbox_obj,
            key_in_world="instance_name",
            bbox_root_key="buildings",
            write_to="properties.bbox",
            default_zero_for_buildings=True,
            also_copy_keys=[]  # 需要也写 num_limit 可改为 ["num_limit"]
        )

    # 5) 保存
    save_json(world, output_path)

    # Logs
    print(
        f"restaurants={len(restaurants)}, stores={len(stores)}, "
        f"rest_areas={len(rest_areas)}, hospitals={len(hospitals)}, car_rentals={len(car_rentals)}, "
        f"road_chargers={added_road_chargers}"
    )
    if bus_routes:
        print(f"bus_routes=1, stops={len(bus_routes[0]['station_ids'])}, route_points={len(bus_routes[0]['path'])}")
    else:
        print("bus_routes=0")
    print(f"✅ Saved to: {output_path}")

# ===== Example =====
if __name__ == "__main__":
    enrich_world_in_place(
        world_json_path=r"D:\LLMDelivery-LJ\Test_Data\test\progen_world.json",
        output_path=r"D:\LLMDelivery-LJ\Test_Data\test\progen_world_enriched.json",
        roads_json_path=r"D:\LLMDelivery-LJ\Test_Data\test\roads.json",
        bbox_json_path=r"D:\LLMDelivery-LJ\SimWorld\data\Bbox.json",

        # 建筑标签数量
        num_restaurants=5,
        num_stores=5,
        num_rest_areas=2,
        num_hospitals=1,
        num_car_rentals=2,

        # 公交参数（避免拥挤）
        bus_route_target_len=2000.0,
        num_bus_stops=10,
        bus_min_gap_m=50.0,            # 站点最小间距（米）
        bus_spacing_jitter_frac=0.2,   # 等距抖动比
        bus_offset_m=10.0,             # 站点离路中心线距离（米）

        # 充电桩（沿路）
        road_chargers=15,
        charger_offset_m=12.0,         # 桩到中心线距离（米）

        # 坐标缩放（米->厘米）
        scale=100.0,
        rng_seed=42
    )
