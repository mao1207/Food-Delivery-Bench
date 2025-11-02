# -*- coding: utf-8 -*-
"""
stats_aggregate.py  —— 只统计，不做任何动作名处理
递归扫描 INPUT_DIR 下所有 .json，聚合:
- meta.model -> model 名（仅做 basename 清洗，若不需要可改 tidy_model_name 直接返回原串）
- actions.*.{attempts, successes}

输出 JSON：OUT_JSON
"""

import os
import json
from collections import defaultdict, Counter

# ========= 固定路径（按需改你的本地路径） =========
INPUT_DIR = r"D:\BaiduNetdiskDownload\Food-Delivery-Bench-2.0\initial_results"
OUT_JSON  = r"D:\BaiduNetdiskDownload\Food-Delivery-Bench-2.0\draw\figures\actions_stats.json"
# ============================================

def tidy_model_name(raw: str) -> str:
    if raw is None:
        return "unknown"
    s = str(raw)
    if "/" in s:
        s = s.split("/")[-1]
    return s

def iter_json_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".json"):
                yield os.path.join(dirpath, fn)

def aggregate(input_dir: str):
    attempts = defaultdict(lambda: defaultdict(int))
    successes = defaultdict(lambda: defaultdict(int))
    action_global_counter = Counter()
    model_totals = Counter()

    files = list(iter_json_files(input_dir))
    if not files:
        raise RuntimeError(f"No JSON files found under: {input_dir}")

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Skip {path}: {e}")
            continue

        model_raw = (data.get("meta", {}) or {}).get("model", None)
        model = tidy_model_name(model_raw)  # 如不需清洗可改为：model = str(model_raw)

        acts = (data.get("actions", {}) or {})
        for action_key, v in acts.items():
            a = action_key  # **原样使用动作名：不映射、不改大小写**
            att = int((v or {}).get("attempts", 0) or 0)
            suc = int((v or {}).get("successes", 0) or 0)
            suc = min(suc, att)

            attempts[model][a]  += att
            successes[model][a] += suc
            action_global_counter[a] += att
            model_totals[model] += att

    models_sorted  = [m for m, _ in model_totals.most_common()]
    actions_sorted = [a for a, _ in action_global_counter.most_common()]
    return models_sorted, actions_sorted, attempts, successes, action_global_counter

def dump_stats_json(models, actions, attempts, successes, action_counter, input_dir, out_json_path):
    stats = {
        "input_dir": input_dir,
        "actions": list(actions),          # 原始动作名，原样输出
        "models": list(models),
        "global": {
            "action_totals": {a: int(action_counter.get(a, 0)) for a in actions},
            "models_sorted_by_attempts": list(models),
        },
        "per_model": {}
    }
    for m in models:
        row_total_attempts  = int(sum(int(attempts[m].get(a, 0))  for a in actions))
        row_total_successes = int(sum(int(successes[m].get(a, 0)) for a in actions))
        row_total_attempts_safe = row_total_attempts if row_total_attempts > 0 else 1
        actions_block = {}
        for a in actions:
            att = int(attempts[m].get(a, 0))
            suc = int(successes[m].get(a, 0))
            rate = (suc / att) if att > 0 else None
            share = att / row_total_attempts_safe
            actions_block[a] = {
                "attempts": att,
                "successes": suc,
                "success_rate": rate,
                "share_in_row": share,
            }
        stats["per_model"][m] = {
            "total_attempts": row_total_attempts,
            "total_successes": row_total_successes,
            "actions": actions_block
        }

    folder = os.path.dirname(os.path.abspath(out_json_path))
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved stats JSON to: {out_json_path}")

if __name__ == "__main__":
    models, actions, attempts, successes, action_counter = aggregate(INPUT_DIR)
    dump_stats_json(models, actions, attempts, successes, action_counter, INPUT_DIR, OUT_JSON)