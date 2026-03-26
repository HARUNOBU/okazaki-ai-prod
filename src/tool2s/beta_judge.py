import argparse
import json
from pathlib import Path

def judge(file_path: str, expect: str):
    data = json.load(open(file_path, encoding="utf-8"))
    ok = sum(1 for r in data if r.get("status") == expect)
    print(file_path, "=", ok, "/", len(data))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", default="suite_fixed_50_result.json")
    ap.add_argument("--volatile", default="suite_volatile_50_result.json")
    ap.add_argument("--general", default="suite_general_50_result.json")
    args = ap.parse_args()
    judge(args.fixed, "ok")
    judge(args.volatile, "abstain")
    judge(args.general, "ok")
