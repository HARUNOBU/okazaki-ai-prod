# tools/analyze_abstain_normal60.py
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
import csv

# ====== 設定（必要に応じて変更） ======
LOG_PATH_CANDIDATES = [
    Path("out/qa_log.jsonl"),
    Path("logs/qa_log.jsonl"),
    Path("out/answered_log.jsonl"),
]
OUT_CSV = Path("out/abstain_review_normal60.csv")

# 「通常60」の識別に使えるキーがログにあるならここで指定
# 例: run_name / suite / test_set / eval_name など
NORMAL60_KEYS = ["normal60", "通常60", "suite_normal60"]

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # 壊れ行はスキップ
                continue
    return rows

def is_normal60(rec: dict) -> bool:
    # ログ構造が不明なので「どこかのフィールドに normal60 が含まれる」方式で拾う
    blob = json.dumps(rec, ensure_ascii=False)
    return any(k in blob for k in NORMAL60_KEYS)

def is_abstain(rec: dict) -> bool:
    # あなたのログ項目に合わせて増やしてください
    # 例: rec["final"]["type"] == "abstain" / rec["abstain_hit"] / answered_by=="abstain"
    if rec.get("abstain_hit") is True:
        return True
    if (rec.get("answered_by") or "").lower() == "abstain":
        return True
    if (rec.get("final_answer_type") or "").lower() == "abstain":
        return True

    # finalフィールドがあるケース
    final = rec.get("final") or {}
    if isinstance(final, dict):
        if (final.get("type") or "").lower() == "abstain":
            return True

    # 出力本文に "abstain" 文字があるだけの誤検知は避けたいので最後の手段
    return False

def pick_log_path() -> Path:
    for p in LOG_PATH_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(f"Log file not found. tried: {LOG_PATH_CANDIDATES}")

def extract_top_hits(rec: dict, n: int = 3) -> list[dict]:
    """
    検索ヒット（rank/dist/title/url/page等）がログ内にある前提で拾う。
    取れない場合は空でOK。
    """
    # よくある候補キー
    for key in ["hits", "retrieval_hits", "debug_hits", "topk", "candidates"]:
        v = rec.get(key)
        if isinstance(v, list) and v:
            return v[:n]
    # final内にあるケース
    final = rec.get("final") or {}
    if isinstance(final, dict):
        for key in ["hits", "retrieval_hits", "debug_hits"]:
            v = final.get(key)
            if isinstance(v, list) and v:
                return v[:n]
    return []

def main():
    log_path = pick_log_path()
    rows = load_jsonl(log_path)

    normal60 = [r for r in rows if is_normal60(r)]
    abstains = [r for r in normal60 if is_abstain(r)]

    # 6件想定だが、念のため全部出す
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "idx",
        "timestamp",
        "question",
        "category",
        "answered_by",
        "abstain_hit",
        "reason",            # abstain理由があれば
        "top1_dist",
        "top1_title",
        "top1_url",
        "top1_page",
        "top2_dist",
        "top2_title",
        "top2_url",
        "top2_page",
        "top3_dist",
        "top3_title",
        "top3_url",
        "top3_page",
        # 手動レビュー欄
        "判定(FAQ追加/検索改善/正当abstain/質問セット再定義)",
        "根拠あり? (Y/N)",
        "根拠PDF確定(title)",
        "pdf_url",
        "page",
        "根拠メモ(引用したい文の要約)",
        "次アクション(実装タスク)",
    ]

    def g(rec, *keys, default=""):
        for k in keys:
            if k in rec and rec[k] is not None:
                return rec[k]
        return default

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()

        for i, r in enumerate(abstains, start=1):
            hits = extract_top_hits(r, n=3)

            def hit_field(h: dict, k: str):
                if not isinstance(h, dict):
                    return ""
                return h.get(k, "")

            row = {
                "idx": i,
                "timestamp": g(r, "ts", "timestamp", "time", default=""),
                "question": g(r, "question", "q", "user_query", default=""),
                "category": g(r, "category", "router_category", "intent", default=""),
                "answered_by": g(r, "answered_by", default=""),
                "abstain_hit": g(r, "abstain_hit", default=""),
                "reason": g(r, "abstain_reason", "reason", default=""),
            }

            # hits 1..3
            for j in range(3):
                h = hits[j] if j < len(hits) else {}
                row[f"top{j+1}_dist"]  = hit_field(h, "dist")
                row[f"top{j+1}_title"] = hit_field(h, "title")
                row[f"top{j+1}_url"]   = hit_field(h, "url") or hit_field(h, "pdf_url")
                row[f"top{j+1}_page"]  = hit_field(h, "page")

            w.writerow(row)

    print(f"[OK] log_path={log_path}")
    print(f"[OK] normal60_records={len(normal60)} abstains={len(abstains)}")
    print(f"[OK] wrote: {OUT_CSV}")

if __name__ == "__main__":
    main()
