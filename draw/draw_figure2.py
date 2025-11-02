# -*- coding: utf-8 -*-
"""
spend_plot_ribbons.py —— 从 spend_stats_aggregate.py 输出 JSON 读取并绘图
- 在这里做“类别映射/合并”（CATEGORY_CANON_MAP）
- 支持自定义 模型顺序/重命名、类别顺序/重命名
- 仅单层条带（占比），给非零项目最小可视高度
"""

import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from itertools import cycle
from matplotlib.legend import Legend
from collections import defaultdict, Counter

# ===== 固定路径 =====
STATS_JSON = r"D:\BaiduNetdiskDownload\Food-Delivery-Bench-2.0\draw\figures\spend_dist_stats.json"
OUT_PNG    = r"D:\BaiduNetdiskDownload\Food-Delivery-Bench-2.0\draw\figures\spend_dist.png"

# ===== 画布与样式 =====
FIGSIZE      = (24, 12)
DPI          = 300
AXLABEL_SIZE = 30
TICK_SIZE    = 42
XTICK_ROT    = 40
MAX_BAR_HEIGHT  = 0.85
MIN_VISUAL_FRAC = 0.08
TOP_K_CATS      = 20  # 当未指定 CATEGORY_ORDER 时，按全局金额选前 K

# ======= 类别/模型控制（合并、顺序、重命名，仅影响显示或聚合）=======
# 合并映射：左边是 stats 里的“原始类别名”（小写/原样都可以；函数里会 .lower() 匹配）
CATEGORY_CANON_MAP = {
    "purchases": "PURCHASES",
    "purchase":  "PURCHASES",
    "charging":  "CHARGING",
    "charge":    "CHARGING",
    "rental":    "RENTAL",
    "rent":      "RENTAL",
    "hospital":  "HOSPITAL",
    "help_expense": "HELP_EXPENSE",
}
# 横轴顺序（写“映射后”的 canonical 名；留空则走 TOP_K）
CATEGORY_ORDER = [
    "PURCHASES", "CHARGING", "RENTAL", "HOSPITAL", "HELP_EXPENSE"
]
# 横轴显示重命名（仅显示）
CATEGORY_RENAME = {
    "PURCHASES":    "Purchases",
    "CHARGING":     "Charging",
    "RENTAL":       "Car Rental",
    "HOSPITAL":     "Hospital",
    "HELP_EXPENSE": "Help Expense",
}
# 纵轴模型顺序与重命名（仅显示；不影响数据键）
MODEL_ORDER = [
    "llama-3.2-90b-vision-instruct",
    "qwen2.5-vl-72b-instruct",
    "gemini-2.5-flash",
    "claude-3.7-sonnet",
    "gpt-4o",
    "gpt-5",
]

MODEL_RENAME = {
    "gpt-5":                         "GPT-5",
    "gpt-4o":                        "GPT-4o",
    "claude-3.7-sonnet":             "Claude-3.7-Sonnet",
    "gemini-2.5-flash":              "Gemini-2.5-Flash",
    "qwen2.5-vl-72b-instruct":       "Qwen2.5-VL-72B-Ins",
    "llama-3.2-90b-vision-instruct": "LLaMA-3.2-90B-Vision-Ins",
}

# ===== 工具函数 =====
def _resolve_out(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    return path

def ribbon_bar_polygon(x_points, y_base, heights, scale_y=1.0):
    x_points = np.asarray(x_points, float)
    h = np.asarray(heights, float)
    if len(x_points) == 1:
        half = 0.5
        xs = np.array([x_points[0]-half, x_points[0]+half, x_points[0]+half, x_points[0]-half])
        ys = np.array([y_base, y_base, y_base+scale_y*h[0], y_base+scale_y*h[0]])
        return xs, ys
    mids = (x_points[:-1] + x_points[1:]) / 2.0
    left  = x_points[0]  - (mids[0] - x_points[0])
    right = x_points[-1] + (x_points[-1] - mids[-1])
    edges = np.concatenate([[left], mids, [right]])
    xs, ys = [], []
    for i in range(len(x_points)):
        xs.extend([edges[i], edges[i+1]]); ys.extend([y_base, y_base])
    for i in reversed(range(len(x_points))):
        xs.extend([edges[i+1], edges[i]]); ys.extend([y_base + scale_y*h[i], y_base + scale_y*h[i]])
    return np.array(xs), np.array(ys)

def load_stats(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        s = json.load(f)
    models = s["models"]
    categories = s["categories"]   # 原始类别名
    pm = s["per_model"]
    # 还原每模型每类别金额
    amounts = {m: {c: float(pm[m]["amounts"].get(c, 0.0)) for c in categories} for m in models}
    return models, categories, amounts

def canon_cat(name: str) -> str:
    if not isinstance(name, str):
        return str(name)
    k = name.strip().lower()
    return CATEGORY_CANON_MAP.get(k, name.upper())  # 未映射则保留原名（转大写更整齐）

def remap_and_accumulate(models, raw_categories, amounts):
    """按 CATEGORY_CANON_MAP 合并类别；返回 canonical 类别、合并后的 per-model 金额、全局 totals。"""
    canon_amounts = {m: defaultdict(float) for m in models}
    canon_totals  = Counter()
    for m in models:
        for c in raw_categories:
            canon = canon_cat(c)
            val = float(amounts[m].get(c, 0.0))
            if val > 0:
                canon_amounts[m][canon] += val
                canon_totals[canon] += val
    canon_categories = [c for c, _ in canon_totals.most_common()]  # 默认按全局金额降序
    return canon_categories, canon_amounts, canon_totals

def resolve_model_order(models, desired):
    if not desired:
        return list(models)
    exist = set(models)
    ordered = [m for m in desired if m in exist]
    for m in models:
        if m not in ordered:
            ordered.append(m)
    return ordered

def resolve_category_order(all_cats, desired, top_k, canon_totals):
    exist = set(all_cats)
    if desired:
        picked = [c for c in desired if c in exist]
        if picked:
            return picked
    # 兜底：按全局金额前 K
    return [c for c, _ in canon_totals.most_common(top_k)]

# ===== 绘图 =====
def plot_and_save(models, cats, amounts, out_png):
    x_pos = np.arange(len(cats), dtype=float)
    y_pos = np.arange(len(models), dtype=float)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    # ax.set_xlabel("Spending Categories", fontsize=AXLABEL_SIZE, labelpad=10)
    # ax.set_ylabel("Models", fontsize=AXLABEL_SIZE, labelpad=10)

    # 横轴显示名
    display_cats = [CATEGORY_RENAME.get(c, c) for c in cats]
    ax.set_xticks(x_pos, display_cats, rotation=XTICK_ROT, ha="right", fontsize=TICK_SIZE)

    # 纵轴显示名
    display_models = [MODEL_RENAME.get(m, m) for m in models]
    ax.set_yticks(y_pos, display_models, fontsize=TICK_SIZE)

    ax.tick_params(axis='both', which='both', length=6, width=1.2)
    ax.set_xlim(x_pos[0]-0.6, x_pos[-1]+0.6)
    ax.set_ylim(-0.2, (len(models)-1) + MAX_BAR_HEIGHT + 0.08)
    ax.margins(y=0)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.grid(axis="y", alpha=0.18, linewidth=0.5)

    cmap = plt.get_cmap("tab20")
    color_cycle = cycle([cmap(i) for i in range(cmap.N)])
    min_h = MAX_BAR_HEIGHT * MIN_VISUAL_FRAC

    for yi, m in enumerate(models):
        y = y_pos[yi]
        color = next(color_cycle)
        vals = np.array([float(amounts[m].get(c, 0.0)) for c in cats], dtype=float)
        row_sum = float(vals.sum()) or 1.0
        shares = vals / row_sum
        heights = MAX_BAR_HEIGHT * shares
        heights = np.where(vals > 0, np.maximum(heights, min_h), 0.0)
        heights = np.minimum(heights, MAX_BAR_HEIGHT)

        xs, ys = ribbon_bar_polygon(x_pos, y_base=y, heights=heights, scale_y=1.0)
        ax.add_patch(Polygon(np.c_[xs, ys], closed=True, linewidth=0.9,
                             edgecolor=color, facecolor=(*color[:3], 0.72), zorder=2))

    # 去掉任何图例
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    for child in ax.get_children():
        if isinstance(child, Legend):
            child.remove()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure to: {out_png}")

# ===== 主流程 =====
if __name__ == "__main__":
    # 1) 读取统计（原始）JSON
    models_raw, categories_raw, amounts_raw = load_stats(STATS_JSON)

    # 2) 类别映射与聚合（绘图阶段）
    cats_all, amounts, cat_totals = remap_and_accumulate(models_raw, categories_raw, amounts_raw)

    # 3) 决定纵轴模型顺序
    models = resolve_model_order(models_raw, MODEL_ORDER)

    # 4) 决定横轴类别顺序
    cats = resolve_category_order(cats_all, CATEGORY_ORDER, TOP_K_CATS, cat_totals)
    if not cats:
        raise ValueError("No categories to plot. Check CATEGORY_CANON_MAP / CATEGORY_ORDER or stats JSON.")

    # 5) 仅保留要画的类别
    amounts_sel = {m: {c: amounts[m].get(c, 0.0) for c in cats} for m in models}

    # 6) 绘图
    plot_and_save(models, cats, amounts_sel, OUT_PNG)