from __future__ import annotations
import json, csv
from pathlib import Path

LOG_PATH = Path("logs/qa_log.jsonl")
OUT_CSV  = Path("out/abstain_all.csv")

def load_jsonl(p: Path):
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: 
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def main():
    rows = load_jsonl(LOG_PATH)

    abst = [r for r in rows if (r.get("status") == "abstain")]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "idx","ts","query","chosen_cat","collection","mode",
        "best_dist","threshold_cat","num_cites",
        "cite1","cite2","cite3","cite4","cite5"
    ]

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()

        for i, r in enumerate(abst, start=1):
            cites = r.get("cites") or []
            th = r.get("thresholds") or {}
            cat = r.get("chosen_cat") or ""
            row = {
                "idx": i,
                "ts": r.get("ts",""),
                "query": r.get("query",""),
                "chosen_cat": cat,
                "collection": r.get("collection",""),
                "mode": r.get("mode",""),
                "best_dist": r.get("best_dist",""),
                "threshold_cat": th.get(cat,""),
                "num_cites": r.get("num_cites",""),
            }
            for k in range(5):
                row[f"cite{k+1}"] = cites[k] if k < len(cites) else ""
            w.writerow(row)

    print(f"[OK] total={len(rows)} abstains={len(abst)} wrote={OUT_CSV}")

if __name__ == "__main__":
    main()
