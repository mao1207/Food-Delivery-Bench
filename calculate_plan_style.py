# -*- coding: utf-8 -*-
"""
Aggregate step-level scores into per-model summaries.

- Scan INPUT_DIR for *.json / *.jsonl files
- Read records shaped like:
  {"model": "...", "step": int, "scores": {risk:int,...}, ...}
- Skip any score == -1
- Output one JSON with per-model metrics

Edit INPUT_DIR and OUT_PATH to your paths.
"""

import os
import json
from typing import Dict, Any, Iterable, Tuple, List, Set

# ====== EDIT THESE ======
INPUT_DIR = r"D:\BaiduNetdiskDownload\Food-Delivery-Bench-2.0-iso\LLM-Delivery\Scripts\debug_snaps\medium-20-scores"
OUT_PATH  = r"D:\BaiduNetdiskDownload\Food-Delivery-Bench-2.0-iso\LLM-Delivery\Scripts\debug_snaps\medium-20-scores\_summary.json"
# ========================

DIMS = ["risk", "long_term", "diversity", "collaboration", "meticulousness", "adaptability"]
EXCLUDE_IN_MACRO = {"collaboration"}  # 宏平均里剔除协作维度（基本都为 -1）

def _safe_iter_file(path: str) -> Iterable[dict]:
    """
    Yield JSON records from a file.
    Supports:
      - jsonl: one JSON object per line
      - json: whole file is [objects...] or a single {object}
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        if not text:
            return
        # Try jsonl first (line by line)
        # If any line fails to parse as JSON, fall back to full-file JSON
        lines = text.splitlines()
        ok_line_mode = True
        recs: List[dict] = []
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    recs.append(obj)
                else:
                    # not a single-object line -> switch to full-file parsing
                    ok_line_mode = False
                    break
            except Exception:
                ok_line_mode = False
                break

        if ok_line_mode and recs:
            for r in recs:
                yield r
            return

        # Fall back to full-file JSON
        try:
            obj = json.loads(text)
        except Exception:
            # Not JSON at all
            return
        if isinstance(obj, list):
            for it in obj:
                if isinstance(it, dict):
                    yield it
        elif isinstance(obj, dict):
            yield obj

def _try_int(x: Any) -> Tuple[bool, int]:
    try:
        return True, int(x)
    except Exception:
        return False, 0

def aggregate_folder(input_dir: str) -> Dict[str, Any]:
    """
    Returns:
      {
        model_name: {
          "metrics": {
             dim: {"mean": float or null, "count": int, "skipped": int, "total_steps_seen": int}
          },
          "macro_avg_excl_collab": float or null,
          "steps_covered": int,
          "files_count": int
        },
        ...
      }
    """
    per_model = {}
    files = [os.path.join(input_dir, fn)
             for fn in os.listdir(input_dir)
             if os.path.isfile(os.path.join(input_dir, fn)) and fn.lower().endswith((".json", ".jsonl"))]

    for fp in files:
        for rec in _safe_iter_file(fp):
            model = rec.get("model")
            scores = (rec.get("scores") or {})
            step = rec.get("step")

            if not isinstance(model, str) or not isinstance(scores, dict):
                continue

            md = per_model.setdefault(model, {
                "metrics": {d: {"sum": 0.0, "count": 0, "skipped": 0, "total_steps_seen": 0} for d in DIMS},
                "steps_seen_set": set(),  # type: Set[int]
                "files_count": 0,
            })
            if isinstance(step, int):
                md["steps_seen_set"].add(step)

            # Update per-dimension stats
            for d in DIMS:
                md["metrics"][d]["total_steps_seen"] += 1
                v = scores.get(d, -1)
                ok, ival = _try_int(v)
                if not ok or ival == -1:
                    md["metrics"][d]["skipped"] += 1
                else:
                    md["metrics"][d]["sum"] += ival
                    md["metrics"][d]["count"] += 1

        # count file processed per model rough estimate later (after loop we can’t know which belonged to which model easily)
        # Instead, we’ll compute files_count by counting unique files that contained at least one record for the model:
    # Re-scan to compute per-model file coverage
    # (Optional; skip if you don't need it)
    files_with_models = {m: set() for m in per_model.keys()}
    for fp in files:
        models_seen_in_file = set()
        for rec in _safe_iter_file(fp):
            model = rec.get("model")
            if isinstance(model, str) and model in per_model:
                models_seen_in_file.add(model)
        for m in models_seen_in_file:
            files_with_models[m].add(fp)

    # finalize summary: means & macro averages
    summary = {}
    for m, md in per_model.items():
        out_metrics = {}
        dim_means_for_macro: List[float] = []

        for d in DIMS:
            s = md["metrics"][d]
            mean = (s["sum"] / s["count"]) if s["count"] > 0 else None
            out_metrics[d] = {
                "mean": round(mean, 4) if mean is not None else None,
                "count": s["count"],
                "skipped": s["skipped"],
                "total_steps_seen": s["total_steps_seen"],
            }
            if d not in EXCLUDE_IN_MACRO and mean is not None:
                dim_means_for_macro.append(mean)

        macro = round(sum(dim_means_for_macro) / len(dim_means_for_macro), 4) if dim_means_for_macro else None

        summary[m] = {
            "metrics": out_metrics,
            "macro_avg_excl_collab": macro,
            "steps_covered": len(md["steps_seen_set"]),
            "files_count": len(files_with_models.get(m, set())),
        }

    return summary

def print_table(summary: Dict[str, Any]) -> None:
    """
    Print a simple table to console: model and each dim mean (skip None), plus macro.
    """
    header_dims = [d for d in DIMS if d != "collaboration"] + ["macro(avg excl collab)"]
    col_names = ["model"] + header_dims
    widths = [max(len(h), 18) for h in col_names]

    def fmt_cell(s: Any, w: int) -> str:
        s = "" if s is None else s
        if isinstance(s, float):
            s = f"{s:.2f}"
        return str(s).ljust(w)

    print("\n" + " | ".join(h.ljust(w) for h, w in zip(col_names, widths)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for model, data in sorted(summary.items(), key=lambda kv: (kv[1]["macro_avg_excl_collab"] or -1), reverse=True):
        row = [model]
        for d in DIMS:
            if d == "collaboration":
                continue
            row.append(data["metrics"][d]["mean"])
        row.append(data["macro_avg_excl_collab"])
        print(" | ".join(fmt_cell(c, w) for c, w in zip(row, widths)))
    print()

def main():
    summary = aggregate_folder(INPUT_DIR)

    # write JSON
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved summary -> {OUT_PATH}")
    print_table(summary)

if __name__ == "__main__":
    main()
