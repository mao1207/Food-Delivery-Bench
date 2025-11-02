# -*- coding: utf-8 -*-
"""
spend_stats_aggregate.py —— 只统计支出分布，不做类别映射
- 递归扫描 INPUT_DIR 下所有 .json
- 优先读取 money.totals.expense_breakdown（原始键原样保留）
- 若 breakdown 缺失，则用 money.details 中的 charging / rental / hospital 估算（使用这些原始键）
- 输出 JSON 包含：models、categories、per_model 数值与占比、全局 totals
"""

import os, json
from collections import defaultdict, Counter

# ===== 固定路径 =====
INPUT_DIR   = r"D:\BaiduNetdiskDownload\Food-Delivery-Bench-2.0\initial_results"
OUT_JSON    = r"D:\BaiduNetdiskDownload\Food-Delivery-Bench-2.0\draw\figures\spend_dist_stats.json"

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

def load_spend_raw(input_dir):
    spend = defaultdict(lambda: defaultdict(float))  # spend[model][raw_category] = amount
    model_totals = Counter()
    cat_totals   = Counter()

    for path in iter_json_files(input_dir):
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue

        model = tidy_model(((d.get("meta") or {}).get("model")))
        totals = ((d.get("money") or {}).get("totals") or {})
        breakdown = (totals.get("expense_breakdown") or {})

        used_any = False
        # 1) 直接用 breakdown（原始键）
        for k, v in breakdown.items():
            try:
                val = float(v) if v is not None else 0.0
            except Exception:
                val = 0.0
            if val > 0:
                spend[model][k] += val
                model_totals[model] += val
                cat_totals[k] += val
                used_any = True

        # 2) 没有 breakdown 时，尽量从 details 估算，键保持原始：charging/rental/hospital
        if not used_any:
            details = ((d.get("money") or {}).get("details") or {})
            for rec in (details.get("charging") or []):
                val = float(rec.get("cost") or 0.0)
                if val > 0:
                    spend[model]["charging"] += val
                    model_totals[model] += val
                    cat_totals["charging"] += val
            for rec in (details.get("rental") or []):
                val = float(rec.get("cost") or 0.0)
                if val > 0:
                    spend[model]["rental"] += val
                    model_totals[model] += val
                    cat_totals["rental"] += val
            for rec in (details.get("hospital") or []):
                val = float(rec.get("fee") or 0.0)
                if val > 0:
                    spend[model]["hospital"] += val
                    model_totals[model] += val
                    cat_totals["hospital"] += val
            # 其他可能的明细就不推断了，保持 0

    models_sorted = [m for m, _ in model_totals.most_common()] or ["unknown"]
    categories_sorted = [c for c, _ in cat_totals.most_common()]
    return models_sorted, categories_sorted, spend, cat_totals

def dump_stats(models, categories, spend, cat_totals, out_json):
    stats = {
        "input_dir": INPUT_DIR,
        "models": models,
        "categories": categories,  # 原始类别名
        "global": {"category_totals": {c: float(cat_totals.get(c, 0.0)) for c in categories}},
        "per_model": {}
    }
    for m in models:
        row = {c: float(spend[m].get(c, 0.0)) for c in categories}
        row_total = sum(row.values())
        denom = row_total if row_total > 0 else 1.0
        stats["per_model"][m] = {
            "amounts": row,
            "row_total": float(row_total),
            "shares": {c: (row[c] / denom) for c in categories}
        }
    with open(_resolve_out(out_json), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved spend stats to: {out_json}")

if __name__ == "__main__":
    models, categories, spend, cat_totals = load_spend_raw(INPUT_DIR)
    dump_stats(models, categories, spend, cat_totals, OUT_JSON)