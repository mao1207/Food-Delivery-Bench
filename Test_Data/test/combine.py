# merge_bbox_into_world.py
# -*- coding: utf-8 -*-
import json
from pathlib import Path

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def get_world_items(world_data):
    """
    world.json 可能是：
      1) 直接是 list[dict]
      2) 是 dict，里面的关键字段是 'nodes' 或其他名字
    这个函数返回可迭代的对象列表及其“容器信息”，以便回写。
    """
    if isinstance(world_data, list):
        return world_data, ("list", None)

    # 常见包裹字段名尝试
    for key in ["nodes", "objects", "items", "actors", "pois"]:
        if key in world_data and isinstance(world_data[key], list):
            return world_data[key], ("dict", key)

    # 兜底：当作空列表处理
    return [], ("unknown", None)

def merge_bbox(world_path, bbox_path, out_path,
               key_in_world="instance_name",
               bbox_root_key="buildings",
               write_to="properties.bbox",
               also_copy_keys=None):
    """
    - world_path: 原 world.json
    - bbox_path:  含 buildings 映射的 bbox.json
    - out_path:   输出合并后的 JSON
    - key_in_world: 在 world 条目中，用于匹配的键（默认 instance_name）
    - bbox_root_key: 在 bbox.json 中保存映射的根键（默认 "buildings"）
    - write_to:   bbox 写入到 world 条目的哪个字段（默认 properties.bbox）
    - also_copy_keys: 还想从 bbox.json 复制的其它键（例如 ["num_limit"]）
    """
    if also_copy_keys is None:
        also_copy_keys = []  # 只复制 bbox

    world = load_json(world_path)
    bbox_data = load_json(bbox_path)

    # 构造 name -> info 的映射
    buildings = bbox_data.get(bbox_root_key, {})
    # 统一结构：每个 value 里应该有 'bbox' 和可能的其他键
    # { "BP_Building_03_C": {"bbox": {...}, "num_limit": 5}, ... }

    items, container = get_world_items(world)
    total = len(items)
    hit = 0
    miss = 0

    def set_nested(holder: dict, dotted_key: str, value):
        """把 value 写到 a.b.c 这种路径下，没有就创建字典"""
        cur = holder
        *parents, last = dotted_key.split(".")
        for k in parents:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        cur[last] = value

    for obj in items:
        name = obj.get(key_in_world)
        if not name:
            miss += 1
            continue

        info = buildings.get(name)
        if not info:
            miss += 1
            continue

        # 写 bbox
        if "bbox" in info:
            set_nested(obj, write_to, info["bbox"])

        # 可选：拷贝额外键
        for k in also_copy_keys:
            if k in info:
                set_nested(obj, f"properties.{k}", info[k])

        hit += 1

    # 回写（对 list/dict 两种容器都兼容）
    if container[0] == "list":
        out_obj = items
    elif container[0] == "dict":
        key = container[1]
        world[key] = items
        out_obj = world
    else:
        # 结构未知，直接把 world 原样保存（若 items 改动则也在其中）
        out_obj = world

    save_json(out_obj, out_path)
    print(f"Merged bbox -> {out_path}")
    print(f"Total items: {total}, matched: {hit}, skipped: {miss}")

if __name__ == "__main__":
    # 把路径改成你的实际文件
    merge_bbox(
        world_path=r"D:\LLMDelivery-LJ\Test_Data\test\progen_world_enriched.json",
        bbox_path=r"D:\LLMDelivery-LJ\SimWorld\data\Bbox.json",
        out_path=r"D:\LLMDelivery-LJ\Test_Data\test\progen_world_bbox.json",
        # 想连同 num_limit 一起写入 properties，可放开下一行：
        # also_copy_keys=["num_limit"],
    )
