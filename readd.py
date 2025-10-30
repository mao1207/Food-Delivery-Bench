#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fix hospital fee from 20 -> 5 in DeliveryBench run_report JSONs (recursive).
修改 hospital 每次费用，从 20 改为 5，同时把多扣的钱补回去。
"""

import json
from pathlib import Path
from typing import Any, Dict

# ====== 在这里写死要处理的文件夹路径 ======
root = Path(r"D:\BaiduNetdiskDownload\Food-Delivery-Bench-2.0\initial_results")

OLD_FEE = 20.0
NEW_FEE = 5.0
DELTA   = OLD_FEE - NEW_FEE  # 15.0

TOTALS_KEYS_TO_UPDATE = [
    ("expense_breakdown", "hospital", -1),   # -= refund
    ("expense_total", None, -1),             # -= refund
    ("current_balance", None, +1),           # += refund
    ("net_growth", None, +1),                # += refund
    ("observed_agent_balance_end", None, +1) # += refund (if present)
]

PER_HOUR_MAP = [
    ("net_growth", "net_growth"),
    ("orders_income", "orders_income"),
    ("help_income", "help_income"),
    ("expense_total", "expense_total"),
    ("base_wage", "base_wage"),
    ("bonus", "bonus"),
    ("initial_earnings", "initial_earnings"),
    ("current_balance", "current_balance"),
    ("help_expense", "help_expense"),
    ("observed_agent_balance_end", "observed_agent_balance_end"),
]

def _get_hours(meta: Dict[str, Any]) -> float:
    if "lifecycle_hours" in meta and meta["lifecycle_hours"] > 0:
        return float(meta["lifecycle_hours"])
    if "active_hours" in meta and meta["active_hours"] > 0:
        return float(meta["active_hours"])
    start = meta.get("started_sim_s")
    end = meta.get("ended_sim_s")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start:
        return (end - start) / 3600.0
    return 2.0

def fix_file(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    
    money = data.get("money", {})
    details = money.get("details", {})
    hospital = details.get("hospital", [])
    if not hospital:
        return None

    fixed_count = 0
    for h in hospital:
        if isinstance(h.get("fee"), (int, float)) and h["fee"] == OLD_FEE:
            h["fee"] = NEW_FEE
            fixed_count += 1
    if fixed_count == 0:
        return None

    refund = DELTA * fixed_count
    totals = money.get("totals", {})

    # 更新 totals
    for key, subkey, direction in TOTALS_KEYS_TO_UPDATE:
        if subkey is None:
            if isinstance(totals.get(key), (int, float)):
                totals[key] = float(totals[key]) + direction * refund
        else:
            eb = totals.get(key, {})
            if isinstance(eb.get(subkey), (int, float)):
                eb[subkey] = float(eb[subkey]) + direction * refund
            else:
                eb[subkey] = direction * refund
            totals[key] = eb
    money["totals"] = totals

    # 更新 per_hour
    hours = _get_hours(data.get("meta", {})) or 2.0
    new_perh = money.get("per_hour", {})
    for per_key, total_key in PER_HOUR_MAP:
        if isinstance(totals.get(total_key), (int, float)):
            new_perh[per_key] = float(totals[total_key]) / hours
    money["per_hour"] = new_perh

    # 写回 hospital details
    details["hospital"] = hospital
    money["details"] = details
    data["money"] = money

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return f"FIXED: {path} | hospital entries: {fixed_count}, refund: {refund:.2f}"

def main():
    if not root.exists():
        print(f"❌ Path not found: {root}")
        return
    modified = 0
    for p in root.rglob("*.json"):
        msg = fix_file(p)
        if msg:
            print(msg)
            modified += 1
    print(f"\n✅ Scanned {root}, modified {modified} files.")

if __name__ == "__main__":
    main()