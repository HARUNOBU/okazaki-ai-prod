import argparse
import os
import pandas as pd
import requests
import json
from pathlib import Path

API_URL = os.getenv("BETA_API_URL", "http://127.0.0.1:8501/ask")

def run(file_path: str):
    df = pd.read_csv(file_path)
    results = []

    for q in df["query"]:
        r = requests.post(API_URL, json={"query": q}, timeout=120)
        r.raise_for_status()
        j = r.json()

        results.append({
            "query": q,
            "status": j.get("status"),
            "answered_by": j.get("answered_by"),
            "collection": j.get("collection_used")
        })

    out = Path(file_path).stem + "_result.json"

    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("saved", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", default=["beta_fixed.csv", "beta_volatile.csv", "beta_general.csv"])
    args = ap.parse_args()
    for fp in args.files:
        run(fp)
