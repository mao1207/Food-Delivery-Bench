#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将含 properties.poi_type 的条目按类型归类，导出形如：
{
  "restaurant": { "GEN_BP_Building_05_C_18": { ... } },
  "car rental": { "GEN_BP_Building_75_C_20": { ... } }
}
"""

import json
from typing import Any, Dict, List

# ======== 可在此修改入参/行为 ========
INPUT_PATH = r"D:\LLMDelivery-LJ\Test_Data\test\progen_world_enriched.json"     # 输入 JSON 文件路径
OUTPUT_PATH = r"D:\LLMDelivery-LJ\Test_Data\test\poi.json"   # 输出 JSON 文件路径
KEEP_ORIGINAL_TYPE = False         # True: 保留原始写法；False: 下划线→空格，小写化
INCLUDE_UNKNOWN_BUCKET = False     # True: 没有 poi_type 的也归到 "unknown"
UNKNOWN_BUCKET_NAME = "unknown"    # 无类型归类名
# ===================================

def normalize_type(t: str) -> str:
    """归一化：下划线变空格 + 去首尾空格 + 小写"""
    return t.replace("_", " ").strip().lower()

def load_items(path: str) -> List[Dict[str, Any]]:
    """加载条目列表：支持顶层为 list 或含 nodes/pois/items 的 dict"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("nodes", "pois", "items"):
            v = data.get(key)
            if isinstance(v, list):
                return v
    raise ValueError("未在输入文件中找到条目列表：顶层应为数组或含 nodes/pois/items 字段的对象")

def group_by_poi_type(
    items: List[Dict[str, Any]],
    keep_original_type: bool = False,
    include_unknown: bool = False,
    unknown_name: str = "unknown",
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for obj in items:
        props = obj.get("properties", {}) or {}
        poi_type = props.get("poi_type")

        if poi_type:
            tkey = poi_type if keep_original_type else normalize_type(poi_type)
        else:
            if not include_unknown:
                continue
            tkey = unknown_name

        bucket = grouped.setdefault(tkey, {})
        item_id = obj.get("id") or obj.get("name") or f"item_{len(bucket)+1}"

        # 精简保留必要字段，方便后续使用
        entry = {
            "id": item_id,
            "instance_name": obj.get("instance_name"),
            "location": props.get("location"),
            "orientation": props.get("orientation"),
            "scale": props.get("scale"),
            "bbox": props.get("bbox"),
        }
        if "segment_assignment" in obj:
            entry["segment_assignment"] = obj["segment_assignment"]

        bucket[item_id] = entry

    return grouped

def main():
    items = load_items(INPUT_PATH)
    grouped = group_by_poi_type(
        items,
        keep_original_type=KEEP_ORIGINAL_TYPE,
        include_unknown=INCLUDE_UNKNOWN_BUCKET,
        unknown_name=UNKNOWN_BUCKET_NAME,
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2, sort_keys=True)

    # 简要反馈
    print(f"读取：{INPUT_PATH}")
    print(f"写出：{OUTPUT_PATH}")
    print(f"类别数：{len(grouped)}")
    for k, v in grouped.items():
        print(f" - {k}: {len(v)} 条")

if __name__ == "__main__":
    main()
