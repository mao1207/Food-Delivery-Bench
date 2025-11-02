# -*- coding: utf-8 -*-
"""
score_model.py — step-level scorer (OpenRouter, no CLI flags)

- 扫描 ROOT_DIR 中 {model}_{step}_{kind}.{ext} 命名的文件：
  kinds ∈ {global.png, local.png, prompt.txt, output.txt}（忽略 fpv）
- 解析 output.json（不合法则跳过该 step）
- 用 gpt-4o（经 OpenRouter）按 6 个维度打分（0–10；未体现为 -1；collaboration 强制 -1）
- 每个 model 输出到 OUT_DIR/{model}_scores.jsonl（每行一个 step 的记录）

依赖：
  - model.py（上一条消息里给你的通用 BaseModel）
  - pip: pillow numpy openai
环境变量：
  - OPENROUTER_API_KEY（建议这样提供）
"""

from __future__ import annotations
import os, re, json, logging
from typing import Dict, Tuple, List, Any, Optional
from PIL import Image
import sys

SIMWORLD_DIR      = r"D:\BaiduNetdiskDownload\Food-Delivery-Bench-2.0-iso\SimWorld"
LLM_DELIVERY_DIR  = r"D:\BaiduNetdiskDownload\Food-Delivery-Bench-2.0-iso\LLM-Delivery"
sys.path.insert(0, SIMWORLD_DIR); sys.path.insert(0, LLM_DELIVERY_DIR)
from llm.score_model import BaseModel


# =========================
# CONFIG（集中修改）
# =========================
ROOT_DIR = r"D:\BaiduNetdiskDownload\Food-Delivery-Bench-2.0-iso\LLM-Delivery\Scripts\debug_snaps\medium-20"                      # 你的文件夹：包含 {model}_{step}_{kind}.{ext}
OUT_DIR  = r"D:\BaiduNetdiskDownload\Food-Delivery-Bench-2.0-iso\LLM-Delivery\Scripts\debug_snaps\medium-20-scores"                    # 输出目录
# OpenRouter 设置（OpenAI SDK 兼容）
BASE_URL = "https://openrouter.ai/api/v1"
API_KEY  = os.getenv("OPENROUTER_KEY", "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")  # 不建议硬编码；必要时可直接写字符串
SCORER_MODEL = "openai/gpt-4o"           # OpenRouter 模型名（可改 openai/gpt-4o-mini 等）
RATE_LIMIT_PER_MIN = 30                  # 简单的 QPS 节流


# =========================
# 文件名解析
# =========================
FNAME_RE = re.compile(
    r"^(?P<model>.+?)_(?P<step>\d+)_(?P<kind>fpv|global|local|prompt|output)\.(?P<ext>png|txt)$",
    re.IGNORECASE
)


def build_system_prompt() -> str:
    """
    评分用系统提示（非穷举示例；允许 LLM 基于当前证据自发判断）。
    """
    return (
        "You are a strict step-level evaluator for a delivery agent. "
        "You will receive: (1) a GLOBAL MAP image, (2) a LOCAL MAP image, and "
        "(3) TEXT containing the agent's prompt (observation/rules/context) and the agent's JSON output for THIS step. "
        "Your job is to assign 0–10 scores for the following behavioral dimensions. "
        "CRITICAL: Only use evidence present in THIS step's materials; do not hallucinate from general knowledge. "
        "If a dimension is not evidenced this step, output -1 for that dimension.\n\n"

        "Scoring dimensions (integers in [0,10], or -1 if not evidenced). "
        "The examples are NON-EXHAUSTIVE; they illustrate possible signals, but you should consider any behavior that reasonably "
        "indicates the dimension in THIS step:\n"
        "1) risk (risk-taking vs conservatism): 10 = clearly aggressive/high-risk choices (e.g., accepts many orders at once despite conflict risk; "
        "prefers far/high-reward routes; proceeds despite low energy/scooter battery when that creates material risk). "
        "0 = very conservative (e.g., strictly one safe order at a time; pre-emptive charging/resting even when not urgent). "
        "Use any other reasonable evidence from the step (e.g., skipping safety to save time, or over-hedging) to judge. \n"
        "2) long_term (long-term planning): 10 = explicit multi-step foresight (e.g., chains destinations, chooses orders that align geographically, "
        "invests in items like battery packs/energy drinks for future benefit, or writes a follow-up plan that clearly looks beyond the next move). "
        "0 = purely myopic/one-step. -1 if there is no sign of planning beyond the immediate action. Consider both the 'reasoning_and_reflection' and 'future_plan'.\n"
        "3) diversity (strategy diversity): 10 = explores non-routine tools/transport/help mechanisms beyond the typical charge→accept→pickup→deliver loop "
        "(e.g., renting a car, taking a bus, buying special items, using novel movement/coordination strategies). "
        "0 = strictly standard routine. -1 if there is no signal either way.\n"
        "4) collaboration: ALWAYS -1 in this single-agent evaluation (ignore incidental mentions).\n"
        "5) meticulousness (attention to operational details/constraints): 10 = shows careful consideration of perishable/temperature-sensitive items, "
        "separates hot/cold foods, plans ice/heat packs appropriately, respects fragile items (e.g., cakes), honors specified drop-off methods "
        "('hand_to_customer' vs 'leave_at_door'), accounts for melting risk, timing windows, etc. "
        "0 = clearly careless or overlooks important constraints. -1 if not evidenced.\n"
        "6) adaptability (flexibility to update plan based on current state/errors): 10 = adapts plans in light of new state (e.g., pivots to charge when noticing low battery, "
        "changes route after an error). 0 = blindly follows stale plan. -1 if no signal.\n\n"

        "Output policy:\n"
        "- Judge ONLY from THIS STEP's text/images (prompt+output). "
        "- If the model output is malformed/missing, the caller will have skipped the step. "
        '- Return JSON ONLY with EXACT keys and integer values: '
        '{"risk": int, "long_term": int, "diversity": int, "collaboration": int, "meticulousness": int, "adaptability": int}. '
        "No extra keys, no prose, no markdown."
    )


def build_user_block(prompt_text: str, output_json_str: str) -> str:
    """
    合并 prompt 与该 step 的 output JSON。
    这里 prompt 可以包含诸如 ### agent_state / ### recent_actions / ### post_action_plan 等文本。
    """
    return (
        "### PROMPT (observation/rules/context)\n" + prompt_text.strip() + "\n\n" +
        "### MODEL_OUTPUT_JSON (for THIS step)\n" + output_json_str.strip() + "\n"
    )


def open_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def group_files(root: str) -> Dict[Tuple[str, int], Dict[str, str]]:
    """
    返回 (model, step) -> {kind: filepath}
    需要 kinds: global, local, prompt, output   （fpv 忽略）
    """
    groups: Dict[Tuple[str, int], Dict[str, str]] = {}
    for name in os.listdir(root):
        m = FNAME_RE.match(name)
        if not m:
            continue
        model = m.group("model")
        step = int(m.group("step"))
        kind = m.group("kind").lower()
        path = os.path.join(root, name)
        key = (model, step)
        groups.setdefault(key, {})
        groups[key][kind] = path
    return groups


def parse_output_json(text: str) -> Optional[dict]:
    """
    从 output.txt 中定位并解析 JSON；失败返回 None。
    """
    text = text.strip()
    try:
        i = text.index("{")
        j = text.rindex("}") + 1
        obj = json.loads(text[i:j])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def score_one_step(
    scorer: BaseModel,
    system_prompt: str,
    global_img_path: str,
    local_img_path: str,
    prompt_text: str,
    output_json_text: str,
) -> Optional[Dict[str, int]]:
    """
    评分一个 step；返回六维分数字典或 None。
    """
    images = [open_image(global_img_path), open_image(local_img_path)]
    user_text = build_user_block(prompt_text, output_json_text)

    resp = scorer.generate(
        system=system_prompt,
        user=user_text,
        images=images,
        max_tokens=200,
        temperature=0.0,
        n=1,
    )

    try:
        data = json.loads(resp)
        # 当前阶段协作固定 -1（双保险）
        data["collaboration"] = -1
        clean = {}
        for k in ["risk", "long_term", "diversity", "collaboration", "meticulousness", "adaptability"]:
            v = int(data.get(k, -1))
            if v < -1: v = -1
            if v > 10: v = 10
            clean[k] = v
        return clean
    except Exception as e:
        logging.error(f"Scorer returned invalid JSON: {e}\nRaw: {resp}")
        return None


def main():
    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not found. Please set it in your environment.")

    os.makedirs(OUT_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    # 构造 LLM 评分器（走 OpenRouter）
    scorer = BaseModel(
        url=BASE_URL,
        api_key=API_KEY,
        model=SCORER_MODEL,
        max_tokens=256,
        temperature=0.0,
        top_p=1.0,
        rate_limit_per_min=RATE_LIMIT_PER_MIN,
        supports_vision=True,
    )
    system_prompt = build_system_prompt()

    groups = group_files(ROOT_DIR)
    per_model: Dict[str, List[Tuple[int, Dict[str, str]]]] = {}
    for (model, step), files in groups.items():
        per_model.setdefault(model, []).append((step, files))

    for model, items in per_model.items():
        items.sort(key=lambda x: x[0])
        out_path = os.path.join(OUT_DIR, f"{model}_scores.jsonl")
        written = 0
        with open(out_path, "w", encoding="utf-8") as fout:
            for step, files in items:
                # 需要四件：global/local/prompt/output
                if not all(k in files for k in ("global", "local", "prompt", "output")):
                    logging.info(f"[skip] {model} step {step}: missing one of global/local/prompt/output")
                    continue

                prompt_text = load_text(files["prompt"])
                output_text = load_text(files["output"])
                output_obj = parse_output_json(output_text)
                if output_obj is None:
                    logging.info(f"[skip] {model} step {step}: output not valid JSON")
                    continue

                scores = score_one_step(
                    scorer=scorer,
                    system_prompt=system_prompt,
                    global_img_path=files["global"],
                    local_img_path=files["local"],
                    prompt_text=prompt_text,
                    output_json_text=json.dumps(output_obj, ensure_ascii=False),
                )
                if scores is None:
                    logging.info(f"[skip] {model} step {step}: scoring failed")
                    continue

                rec = {
                    "model": model,
                    "step": step,
                    "scores": scores,
                    "files": {
                        "global": os.path.basename(files["global"]),
                        "local": os.path.basename(files["local"]),
                        "prompt": os.path.basename(files["prompt"]),
                        "output": os.path.basename(files["output"]),
                    },
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

        logging.info(f"[done] {model}: wrote {written} lines -> {out_path}")


if __name__ == "__main__":
    main()
