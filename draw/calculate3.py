# -*- coding: utf-8 -*-
"""
transport_stats_aggregate.py —— 只统计交通模式分布（不做映射）
- 递归扫描 INPUT_DIR 下所有 .json
- 读取 activity.mode_time_s（按秒），对每个 model 汇总各交通方式总时长
- 输出 JSON 包含：models、modes（原始键）、per_model 数值与占比、全局 totals
"""

import os, json
from collections import defaultdict, Counter

# ===== 固定路径 =====
INPUT_DIR = r"D:\BaiduNetdiskDownload\Food-Delivery-Bench-2.0\initial_results"
OUT_JSON  = r"D:\BaiduNetdiskDownload\Food-Delivery-Bench-2.0\draw\figures\transport_dist_stats.json"

def _resolve_out(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    return path

def tidy_model(raw: str) -> str:
    if not raw:
        return "unknown"
    s = str(raw)
    return s.split("/")[-1] if "/" in s else s

def iter_json_files(root):
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(".json"):
                yield os.path.join(dp, fn)

def load_mode_raw(input_dir):
    modes = defaultdict(lambda: defaultdict(float))  # modes[model][raw_mode] = seconds
    model_totals = Counter()
    mode_totals  = Counter()

    for path in iter_json_files(input_dir):
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue

        model = tidy_model(((d.get("meta") or {}).get("model")))
        mt = ((d.get("activity") or {}).get("mode_time_s") or {})

        for k, v in (mt.items() if isinstance(mt, dict) else []):
            try:
                sec = float(v) if v is not None else 0.0
            except Exception:
                sec = 0.0
            if sec > 0:
                modes[model][k] += sec
                model_totals[model] += sec
                mode_totals[k] += sec

    models_sorted = [m for m, _ in model_totals.most_common()] or ["unknown"]
    modes_sorted  = [m for m, _ in mode_totals.most_common()]
    return models_sorted, modes_sorted, modes, mode_totals

def dump_stats(models, modes, amounts, mode_totals, out_json):
    stats = {
        "input_dir": INPUT_DIR,
        "models": models,
        "modes": modes,  # 原始模式名
        "global": {"mode_totals_seconds": {md: float(mode_totals.get(md, 0.0)) for md in modes}},
        "per_model": {},
    }
    for m in models:
        row = {md: float(amounts[m].get(md, 0.0)) for md in modes}
        row_total = sum(row.values())
        denom = row_total if row_total > 0 else 1.0
        stats["per_model"][m] = {
            "time_seconds": row,
            "row_total_seconds": float(row_total),
            "shares": {md: (row[md] / denom) for md in modes},
        }
    with open(_resolve_out(out_json), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved transport stats to: {out_json}")

if __name__ == "__main__":
    models, modes, amounts, mode_totals = load_mode_raw(INPUT_DIR)
    dump_stats(models, modes, amounts, mode_totals, OUT_JSON)