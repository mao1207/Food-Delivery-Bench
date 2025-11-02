# -*- coding: utf-8 -*-
"""
transport_plot_ribbons.py —— 从 transport_stats_aggregate.py 输出 JSON 读取并绘图
- 在这里做“交通模式合并映射”（MODE_CANON_MAP）
- 支持自定义 模型顺序/重命名、模式顺序/重命名
- 单层条带（按时间占比），非零项目最小可视高度
- 画图风格尽量与你给的 spend_plot_ribbons 一致
"""

import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from itertools import cycle
from matplotlib.legend import Legend
from collections import defaultdict, Counter

# ===== 固定路径 =====
STATS_JSON = r"D:\BaiduNetdiskDownload\Food-Delivery-Bench-2.0\draw\figures\transport_dist_stats.json"
OUT_PNG    = r"D:\BaiduNetdiskDownload\Food-Delivery-Bench-2.0\draw\figures\transport_dist.png"

# ===== 画布与样式（参考 spend_plot_ribbons）=====
FIGSIZE         = (24, 12)
DPI             = 300
AXLABEL_SIZE    = 30
TICK_SIZE       = 42
XTICK_ROT       = 40
MAX_BAR_HEIGHT  = 0.85
MIN_VISUAL_FRAC = 0.08
TOP_K_MODES     = 20  # 当未指定 MODE_ORDER 时，按全局时间排序取前 K

# ======= 模式/模型控制（合并、顺序、重命名）=======
# 合并映射：左边是 stats 里的“原始模式名”（会 .lower() 匹配）
MODE_CANON_MAP = {
    "walk":        "WALK",
    "walking":     "WALK",
    "e-scooter":   "E-SCOOTER",
    "escooter":    "E-SCOOTER",
    "e_scooter":   "E-SCOOTER",
    "drag_scooter":"DRAG_SCOOTER",
    "bike":        "BIKE",
    "bicycle":     "BIKE",
    "car":         "CAR",
    "bus":         "BUS",
}
# 横轴顺序（写“映射后”的 canonical 名；留空则走 TOP_K）
MODE_ORDER = [
    "WALK", "E-SCOOTER", "DRAG_SCOOTER", "BIKE", "CAR", "BUS",
]
# 横轴显示重命名（仅显示）
MODE_RENAME = {
    "WALK":         "Walk",
    "E-SCOOTER":    "E-scooter",
    "DRAG_SCOOTER": "Drag",
    "BIKE":         "Bike",
    "CAR":          "Car",
    "BUS":          "Bus",
}

# 纵轴模型顺序与重命名（与你刚刚的“反过来”一致）
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
    """和 actions/spend 的画法一致：底对齐的分段多边形（连续条带）"""
    x_points = np.asarray(x_points, float)
    h = np.asarray(heights, float)

    if len(x_points) == 1:
        half = 0.5
        xs = np.array([x_points[0]-half, x_points[0]+half, x_points[0]+half, x_points[0]-half])
        ys = np.array([y_base, y_base, y_base + scale_y*h[0], y_base + scale_y*h[0]])
        return xs, ys

    mids = (x_points[:-1] + x_points[1:]) / 2.0
    left  = x_points[0]  - (mids[0] - x_points[0])
    right = x_points[-1] + (x_points[-1] - mids[-1])
    edges = np.concatenate([[left], mids, [right]])

    xs, ys = [], []
    # 底边（左到右）
    for i in range(len(x_points)):
        xs.extend([edges[i], edges[i+1]]); ys.extend([y_base, y_base])
    # 顶边（右到左）
    for i in reversed(range(len(x_points))):
        xs.extend([edges[i+1], edges[i]]); ys.extend([y_base + scale_y*h[i], y_base + scale_y*h[i]])
    return np.array(xs), np.array(ys)

def load_stats(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        s = json.load(f)
    models = s["models"]
    modes  = s["modes"]   # 原始模式名
    pm     = s["per_model"]
    amounts = {m: {md: float(pm[m]["time_seconds"].get(md, 0.0)) for md in modes} for m in models}
    return models, modes, amounts

def canon_mode(name: str) -> str:
    if not isinstance(name, str):
        return str(name)
    k = name.strip().lower()
    return MODE_CANON_MAP.get(k, name.upper())  # 未映射则保留原名（转大写更整齐）

def remap_and_accumulate(models, raw_modes, amounts):
    """按 MODE_CANON_MAP 合并模式；返回 canonical 模式列表、合并后的 per-model 时间、全局 totals。"""
    canon_amounts = {m: defaultdict(float) for m in models}
    canon_totals  = Counter()
    for m in models:
        for md in raw_modes:
            canon = canon_mode(md)
            sec = float(amounts[m].get(md, 0.0))
            if sec > 0:
                canon_amounts[m][canon] += sec
                canon_totals[canon] += sec
    canon_modes = [md for md, _ in canon_totals.most_common()]  # 全局时长降序
    return canon_modes, canon_amounts, canon_totals

def resolve_model_order(models, desired):
    if not desired:
        return list(models)
    exist = set(models)
    ordered = [m for m in desired if m in exist]
    for m in models:
        if m not in ordered:
            ordered.append(m)
    return ordered

def resolve_mode_order(all_modes, desired, top_k, canon_totals):
    exist = set(all_modes)
    if desired:
        picked = [md for md in desired if md in exist]
        if picked:
            return picked
    # 兜底：按全局时长前 K
    return [md for md, _ in canon_totals.most_common(top_k)]

# ===== 绘图 =====
def plot_and_save(models, modes, amounts, out_png):
    x_pos = np.arange(len(modes), dtype=float)
    y_pos = np.arange(len(models), dtype=float)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    # 和你给的 spend_plot_ribbons 风格一致：不额外显示轴标题
    # ax.set_xlabel("Transport Modes (by time share)", fontsize=AXLABEL_SIZE, labelpad=10)
    # ax.set_ylabel("Models", fontsize=AXLABEL_SIZE, labelpad=10)

    # 横轴显示名
    display_modes = [MODE_RENAME.get(md, md) for md in modes]
    ax.set_xticks(x_pos, display_modes, rotation=XTICK_ROT, ha="right", fontsize=TICK_SIZE)

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

        secs = np.array([float(amounts[m].get(md, 0.0)) for md in modes], dtype=float)
        row_sum = float(secs.sum()) or 1.0

        shares  = secs / row_sum
        heights = MAX_BAR_HEIGHT * shares
        heights = np.where(secs > 0, np.maximum(heights, min_h), 0.0)
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
    models_raw, modes_raw, amounts_raw = load_stats(STATS_JSON)

    # 2) 模式映射与聚合（绘图阶段）
    modes_all, amounts_canon, mode_totals = remap_and_accumulate(models_raw, modes_raw, amounts_raw)

    # 3) 决定纵轴模型顺序
    models = resolve_model_order(models_raw, MODEL_ORDER)

    # 4) 决定横轴模式顺序
    modes = resolve_mode_order(modes_all, MODE_ORDER, TOP_K_MODES, mode_totals)
    if not modes:
        raise ValueError("No transport modes to plot. Check MODE_CANON_MAP / MODE_ORDER or stats JSON.")

    # 5) 仅保留要画的模式
    amounts_sel = {m: {md: amounts_canon[m].get(md, 0.0) for md in modes} for m in models}

    # 6) 绘图
    plot_and_save(models, modes, amounts_sel, OUT_PNG)