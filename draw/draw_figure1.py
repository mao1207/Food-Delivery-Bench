# -*- coding: utf-8 -*-
"""
plot_ribbons.py —— 从 stats_aggregate.py 的输出 JSON 读数据并绘图
功能：
- 可显式指定 纵轴的模型顺序 MODEL_ORDER
- 可显式指定 横轴的动作顺序 ACTION_ORDER（基于映射后的 canonical 名）
- 可对横轴标签进行重命名 ACTION_RENAME（仅影响显示）
- 可对纵轴模型名进行重命名 MODEL_RENAME（仅影响显示）
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from itertools import cycle
from matplotlib.legend import Legend
from collections import defaultdict, Counter

# ========= 固定路径 =========
STATS_JSON = r"D:\BaiduNetdiskDownload\Food-Delivery-Bench-2.0\draw\figures\actions_stats.json"
OUT_PNG    = r"D:\BaiduNetdiskDownload\Food-Delivery-Bench-2.0\draw\figures\actions.png"
# ==========================

# ====== 映射只在绘图阶段进行（未映射的保留原始名）======
ACTION_CANON_MAP = {
    # 核心合并
    "move_to": "MOVE", "step_forward": "MOVE", "turn_around": "MOVE", "wait": "WAIT",
    "pickup": "PICKUP",
    "place_food_in_bag": "PACK",
    "drop_off": "DROP_OFF",
    "accept_order": "ACCEPT_ORDER",
    "view_orders": "VIEW",
    "charge_escooter": "CHARGE",
    "switch_transport": "SWITCH_TRANSPORT",
    "rest": "REST",
    "buy": "BUY",
    "say": "TALK",
    "view_bag": "VIEW_BAG",
    "view_help_board": "VIEW_HELP_BOARD",
    "place_temp_box": "PLACE_TEMP_BOX",
    "use_battery_pack": "USE_BATTERY_PACK",
    "use_energy_drink": "USE_ENERGY_DRINK",
    "use_ice_pack": "USE_ICE_PACK",
    "view_bus_schedule": "VIEW_BUS_SCHEDULE",
    "board_bus": "BOARD_BUS",
    "rent_car": "RENT_CAR",
    "post_help_request": "POST_HELP",
}

# ====== 你想要的顺序与重命名（可为空） ======
# 1) 纵轴：模型顺序（不区分大小写，内部按名字直接匹配）
MODEL_ORDER = [
    "llama-3.2-90b-vision-instruct",
    "qwen2.5-vl-72b-instruct",
    "gemini-2.5-flash",
    "claude-3.7-sonnet",
    "gpt-4o",
    "gpt-5",
]

# 2) 横轴：动作顺序（映射后的 canonical 名；若为空则走 TOP_K 或全量）
ACTION_ORDER = [
    "MOVE", "PICKUP", "PACK", "DROP_OFF",
    "ACCEPT_ORDER", "VIEW", "WAIT",
    "CHARGE", "SWITCH_TRANSPORT",
    "REST", "BUY",
    "TALK", "VIEW_BAG", "VIEW_HELP_BOARD",
    "PLACE_TEMP_BOX", "USE_BATTERY_PACK", "USE_ENERGY_DRINK", "USE_ICE_PACK",
    "VIEW_BUS_SCHEDULE", "BOARD_BUS", "RENT_CAR",
]

# 3) 横轴显示重命名（仅影响显示标签，不影响数据计算）
ACTION_RENAME = {
    "MOVE": "Move/Step",
    "PICKUP": "Pick-up",
    "PACK": "Bag-Pack",
    "DROP_OFF": "Drop-off",
    "ACCEPT_ORDER": "Accept Order",
    "VIEW": "View Orders",
    "WAIT": "Wait",
    "CHARGE": "Charge e-scooter",
    "SWITCH_TRANSPORT": "Switch Transport",
    "REST": "Rest",
    "BUY": "Buy",
    "TALK": "Talk",
    "VIEW_BAG": "View Bag",
    "VIEW_HELP_BOARD": "View Help Board",
    "PLACE_TEMP_BOX": "Place Temp Box",
    "USE_BATTERY_PACK": "Use Battery Pack",
    "USE_ENERGY_DRINK": "Use Energy Drink",
    "USE_ICE_PACK": "Use Ice Pack",
    "VIEW_BUS_SCHEDULE": "View Bus Schedule",
    "BOARD_BUS": "Board Bus",
    "RENT_CAR": "Rent Car",
}

# 4) 纵轴模型显示重命名（仅影响标签显示，不影响数据/顺序）
# 纵轴模型显示重命名（仅影响显示）
MODEL_RENAME = {
    "gpt-5":                         "GPT-5",
    "gpt-4o":                        "GPT-4o",
    "claude-3.7-sonnet":             "Claude-3.7-Sonnet",
    "gemini-2.5-flash":              "Gemini-2.5-Flash",
    "qwen2.5-vl-72b-instruct":       "Qwen2.5-VL-72B-Ins",
    "llama-3.2-90b-vision-instruct": "LLaMA-3.2-90B-Vision-Ins",
}


# 若不手动给顺序：
MANUAL_ACTIONS = None   # 例如: ["MOVE","VIEW","BUY",...]
TOP_K_ACTIONS  = 100    # 当 ACTION_ORDER 与 MANUAL_ACTIONS 都为空时，按全局 attempts 选前 K

# 画布与字号
FIGSIZE      = (36, 12)
DPI          = 300
AXLABEL_SIZE = 30
TICK_SIZE    = 32
XTICK_ROT    = 50

# 条带参数
MAX_BAR_HEIGHT  = 0.85
INNER_SHRINK    = 0.95
MIN_VISUAL_FRAC = 0.08

# ============== 工具函数们 ==============

def canon_action_for_plot(raw_action: str) -> str:
    if not isinstance(raw_action, str):
        return str(raw_action)
    k = raw_action.strip().lower()
    return ACTION_CANON_MAP.get(k, raw_action)

def ribbon_bar_polygon(x_points, y_base, heights, scale_y=1.0):
    x_points = np.asarray(x_points, dtype=float)
    h = np.asarray(heights, dtype=float)

    if len(x_points) == 1:
        half = 0.5
        xs = np.array([x_points[0]-half, x_points[0]+half,
                       x_points[0]+half, x_points[0]-half])
        ys = np.array([y_base, y_base, y_base + scale_y*h[0], y_base + scale_y*h[0]])
        return xs, ys

    mids = (x_points[:-1] + x_points[1:]) / 2.0
    left_edge  = x_points[0]  - (mids[0] - x_points[0])
    right_edge = x_points[-1] + (x_points[-1] - mids[-1])
    edges = np.concatenate([[left_edge], mids, [right_edge]])

    xs, ys = [], []
    for i in range(len(x_points)):  # 底边
        xs.extend([edges[i], edges[i+1]])
        ys.extend([y_base, y_base])
    for i in reversed(range(len(x_points))):  # 顶边
        xs.extend([edges[i+1], edges[i]])
        ys.extend([y_base + scale_y*h[i], y_base + scale_y*h[i]])
    return np.array(xs), np.array(ys)

def load_stats(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    models    = stats["models"]
    actions   = stats["actions"]                 # 原始动作名列表
    per_model = stats["per_model"]
    attempts  = {m: {a: per_model[m]["actions"][a]["attempts"]  for a in actions} for m in models}
    successes = {m: {a: per_model[m]["actions"][a]["successes"] for a in actions} for m in models}
    return models, actions, attempts, successes

def remap_and_accumulate(models, raw_actions, attempts, successes):
    """将原始动作名按 ACTION_CANON_MAP 合并；未映射的保持原名"""
    canon_attempts = {m: defaultdict(int) for m in models}
    canon_successes= {m: defaultdict(int) for m in models}
    canon_global_counter = Counter()

    for m in models:
        for a in raw_actions:
            canon = canon_action_for_plot(a)
            att = int(attempts[m].get(a, 0))
            suc = int(successes[m].get(a, 0))
            canon_attempts[m][canon]  += att
            canon_successes[m][canon] += min(suc, att)
            canon_global_counter[canon] += att

    canon_actions = [a for a, _ in canon_global_counter.most_common()]
    return canon_actions, canon_attempts, canon_successes, canon_global_counter

def resolve_model_order(models, desired_order):
    """根据用户给的 MODEL_ORDER 生成纵轴顺序；未列出的模型按原顺序追加到末尾。"""
    if not desired_order:
        return list(models)
    desired = list(desired_order)
    exist_set = set(models)  # 只保留存在于数据中的名字
    ordered = [m for m in desired if m in exist_set]
    for m in models:         # 追加剩余未指定的模型（保持原 models 顺序）
        if m not in ordered:
            ordered.append(m)
    return ordered

def resolve_action_order(all_actions, desired_order, manual_actions, top_k, canon_counter):
    """
    决定横轴顺序（映射后的动作名）。
    - 优先使用 ACTION_ORDER 与数据的交集（保持顺序）
    - 若交集为空，尝试 MANUAL_ACTIONS
    - 若仍为空，则退回 top_k 最常见动作
    """
    exist = set(all_actions)
    if desired_order:
        picked = [a for a in desired_order if a in exist]
        if picked:
            return picked
    if manual_actions:
        picked = [a for a in manual_actions if a in exist]
        if picked:
            return picked
    return [a for a, _ in canon_counter.most_common(top_k)]

def plot_ribbons(models, actions, attempts, successes, out_png, action_labels=None):
    x_pos = np.arange(len(actions), dtype=float)
    y_pos = np.arange(len(models))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    # 如需轴标题，取消注释：
    # ax.set_xlabel("Actions", fontsize=AXLABEL_SIZE, fontweight="bold", labelpad=10)
    # ax.set_ylabel("Models",  fontsize=AXLABEL_SIZE, fontweight="bold", labelpad=10)

    # 横轴标签（动作）重命名
    display_names = [action_labels.get(a, a) for a in actions] if action_labels else actions
    ax.set_xticks(x_pos, display_names, rotation=XTICK_ROT, ha="right", fontsize=TICK_SIZE)

    # 纵轴标签（模型）重命名
    display_models = [MODEL_RENAME.get(m, m) for m in models]
    ax.set_yticks(y_pos, display_models, fontsize=TICK_SIZE)

    # 刻度样式
    ax.tick_params(axis='both', which='both', length=6, width=1.2)

    ax.set_xlim(x_pos[0]-0.6, x_pos[-1]+0.6)
    ax.set_ylim(-0.2, (len(models)-1) + MAX_BAR_HEIGHT + 0.10)
    ax.margins(y=0)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.grid(axis="y", alpha=0.18, linewidth=0.5)

    cmap = plt.get_cmap("tab20")
    color_cycle = cycle([cmap(i) for i in range(cmap.N)])

    min_h = MAX_BAR_HEIGHT * MIN_VISUAL_FRAC

    for yi, m in enumerate(models):
        y = y_pos[yi]
        base_color = next(color_cycle)

        row_total = sum(int(attempts[m].get(a, 0)) for a in actions) or 1
        att_vec   = np.array([int(attempts[m].get(a, 0)) for a in actions], dtype=float)
        shares    = att_vec / row_total

        outer_h = MAX_BAR_HEIGHT * shares
        outer_h = np.where(att_vec > 0, np.maximum(outer_h, min_h), 0.0)
        outer_h = np.minimum(outer_h, MAX_BAR_HEIGHT)

        with np.errstate(divide="ignore", invalid="ignore"):
            succ_rate = np.where(att_vec > 0, ([
                (int(successes[m].get(a, 0))/int(attempts[m].get(a, 0)))
                if int(attempts[m].get(a, 0)) > 0 else 0.0 for a in actions
            ]), 0.0).astype(float)
        succ_rate = np.clip(succ_rate, 0.0, 1.0)
        inner_h = outer_h * succ_rate * INNER_SHRINK

        xs, ys = ribbon_bar_polygon(x_pos, y_base=y, heights=outer_h, scale_y=1.0)
        ax.add_patch(Polygon(np.c_[xs, ys], closed=True, linewidth=0.9,
                             edgecolor=base_color, facecolor=(*base_color[:3], 0.28), zorder=2))
        xs_i, ys_i = ribbon_bar_polygon(x_pos, y_base=y, heights=inner_h, scale_y=1.0)
        ax.add_patch(Polygon(np.c_[xs_i, ys_i], closed=True, linewidth=0.0,
                             facecolor=(*base_color[:3], 0.85), zorder=3))

    # 无图例
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    for child in ax.get_children():
        if isinstance(child, Legend):
            child.remove()

    folder = os.path.dirname(os.path.abspath(out_png))
    if folder:
        os.makedirs(folder, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure to: {out_png}")

# ================== 主流程 ==================
if __name__ == "__main__":
    # 1) 读取统计 JSON
    models, raw_actions, attempts_raw, successes_raw = load_stats(STATS_JSON)

    # 2) 映射并合并动作
    canon_actions_all, attempts_canon, successes_canon, canon_counter = remap_and_accumulate(
        models, raw_actions, attempts_raw, successes_raw
    )

    # 3) 决定纵轴模型顺序
    models_ordered = resolve_model_order(models, MODEL_ORDER)
    # 只保留存在于 attempts_canon 的模型
    models_ordered = [m for m in models_ordered if m in attempts_canon]

    # 4) 决定横轴动作顺序（基于“映射后”的名字）
    actions_ordered = resolve_action_order(
        canon_actions_all, ACTION_ORDER, MANUAL_ACTIONS, TOP_K_ACTIONS, canon_counter
    )
    if not actions_ordered:
        raise ValueError("No actions to plot. Check ACTION_CANON_MAP/ACTION_ORDER or stats JSON.")

    # 5) 筛选数据到最终集合
    attempts_sel  = {m: {a: attempts_canon[m].get(a, 0) for a in actions_ordered} for m in models_ordered}
    successes_sel = {m: {a: successes_canon[m].get(a, 0) for a in actions_ordered} for m in models_ordered}

    # 6) 横轴显示名重命名（仅影响标签）
    action_labels = ACTION_RENAME  # 可直接传空 dict

    # 7) 绘图
    plot_ribbons(models_ordered, actions_ordered, attempts_sel, successes_sel, OUT_PNG, action_labels)