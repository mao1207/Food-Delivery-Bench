# fix_per_hour_to_two_hours.py
# -*- coding: utf-8 -*-

"""
把每个 JSON 里的“按小时(per_hour)”指标统一按 2 小时归一化：
- money.per_hour.* = money.totals.* / 2.0（存在即改）
- activity.orders_per_hour = orders.completed_count / 2.0（存在即改）

会递归处理 INPUT_DIR 下所有 .json，并把修改后的文件写入 OUTPUT_DIR，保持原有子目录结构。
"""

import os
import json

TARGET_HOURS = 2.0

# 直接写死路径
INPUT_DIR  = r"D:\BaiduNetdiskDownload\Food-Delivery-Bench-2.0\results\20250924_004159"
OUTPUT_DIR = r"D:\BaiduNetdiskDownload\Food-Delivery-Bench-2.0\results\20250924_004159_per_hour_fixed"

# 这些字段会从 money.totals 拿总值来除以 TARGET_HOURS，写回 money.per_hour
PER_HOUR_FIELDS = [
    "net_growth",
    "orders_income",
    "help_income",
    "expense_total",
    "base_wage",
    "bonus",
    "help_expense",
    "initial_earnings",
    "current_balance",
    "observed_agent_balance_end",
]

def ensure_dirs(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def process_one_json(in_path: str, out_path: str) -> None:
    try:
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] skip (read error) {in_path}: {e}")
        return

    # 1) money.per_hour 重算
    money = data.get("money") or {}
    totals = (money.get("totals") or {})
    per_hour = dict(money.get("per_hour") or {})

    for k in PER_HOUR_FIELDS:
        if k in totals:
            try:
                total_val = float(totals.get(k, 0.0) or 0.0)
            except Exception:
                total_val = 0.0
            per_hour[k] = total_val / TARGET_HOURS

    money["per_hour"] = per_hour
    data["money"] = money

    # 2) activity.orders_per_hour
    orders = data.get("orders") or {}
    activity = data.get("activity") or {}
    if "completed_count" in orders:
        try:
            comp = float(orders.get("completed_count") or 0.0)
        except Exception:
            comp = 0.0
        activity["orders_per_hour"] = comp / TARGET_HOURS
        data["activity"] = activity

    # 写出
    ensure_dirs(out_path)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] write failed {out_path}: {e}")

def main():
    cnt = 0
    for dirpath, _, filenames in os.walk(INPUT_DIR):
        for fn in filenames:
            if not fn.lower().endswith(".json"):
                continue
            in_path = os.path.join(dirpath, fn)
            rel = os.path.relpath(in_path, INPUT_DIR)
            out_path = os.path.join(OUTPUT_DIR, rel)
            process_one_json(in_path, out_path)
            cnt += 1

    print(f"[OK] processed {cnt} JSON files into: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
