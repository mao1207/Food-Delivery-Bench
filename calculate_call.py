#!/usr/bin/env python3
import json
from pathlib import Path
import sys

def avg_successes_recursive(root: str):
    root_path = Path(root)
    if not root_path.is_dir():
        print(f"[ERR] Not a directory: {root}")
        return

    successes = []
    files_scanned = 0

    # 递归遍历所有子目录下的 *.json
    for fp in root_path.rglob("*.json"):
        files_scanned += 1
        try:
            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)
            val = data.get("vlm", {}).get("successes", None)
            if isinstance(val, (int, float)):
                successes.append(float(val))
            else:
                print(f"[SKIP] {fp}: missing vlm.successes")
        except Exception as e:
            print(f"[SKIP] {fp}: {e}")

    if not successes:
        print(f"Scanned {files_scanned} files: no valid vlm.successes found.")
        return

    total = sum(successes)
    avg = total / len(successes)
    print(f"Scanned files       : {files_scanned}")
    print(f"Counted files       : {len(successes)}")
    print(f"Total successes     : {int(total)}")
    print(f"Average per file    : {avg:.3f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python avg_successes_recursive.py <root_folder>")
    else:
        avg_successes_recursive(sys.argv[1])
