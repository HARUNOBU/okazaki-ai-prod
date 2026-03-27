# tools/last60_report.py
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter, defaultdict

PORTAL_ROOT = Path(__file__).resolve().parents[1]  # .../okazaki_waste_rag を基準にしている想定
LOG_PATH = PORTAL_ROOT / "logs" / "qa_log.jsonl"

# =========================
# 3分類ルール（提出用）
# =========================
# A：当日変動（keyword） → tourism_dec_reason == "keyword" で確定
# B：固定情報だがRAG不足 → tourism_is_fixed_info==True かつ status=abstain 以外で根拠無し/弱い（または dist高い）
# C：dist閾値設計問題 → tourism_is_fixed_info==True なのに status=abstain で dist落ち、または main_pass相当で落ち（本来拾いたい）

def classify_row(row: dict) -> tuple[str, str]:
    """
    Returns: (bucket, note)
    bucket: "A" / "B" / "C" / "-"
    """
    cat = row.get("category")
    status = row.get("status")
    used = row.get("collection_used")
    q = row.get("query") or ""

    if cat != "観光":
        return "-", "not_tourism"

    # 観光判定の内訳
    dec_reason = row.get("tourism_dec_reason")
    is_fixed = bool(row.get("tourism_is_fixed_info"))
    best_dist = row.get("main_best_dist")
    th = row.get("dist_threshold_main")

    # A：当日変動（keyword abstain）
    if dec_reason == "keyword" and status == "abstain" and used == "tourism_abstain_rule":
        return "A", "keyword(realtime)"

    # 固定情報系
    if is_fixed:
        # C：固定情報なのに dist で tourism_abstain_rule に落ちた（設計問題）
        if status == "abstain" and used == "tourism_abstain_rule":
            return "C", f"fixed_but_abstained dist={best_dist} th={th}"
        # B：固定情報を拾い上げようとして main に行ったが、根拠が弱い/不足（RAG不足）
        # ※ status=abstain でも used=main の場合は、main側の根拠不足が原因なのでBに入れる
        if status == "abstain" and used == "main":
            return "B", f"fixed_but_main_abstain dist={best_dist} th={th}"
        # ok なら分類外
        return "-", "fixed_ok_or_not_abstained"

    # 固定情報でない（一般観光）で dist落ち：基本はB（データ不足）扱い
    if status == "abstain":
        return "B", f"nonfixed_abstain dist={best_dist} th={th}"

    return "-", "ok"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"qa_log.jsonl not found: {path}")
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def main():
    rows = read_jsonl(LOG_PATH)

    # last60（末尾60件）
    last = rows[-60:] if len(rows) >= 60 else rows
    print(f"last60 total: {len(last)}")

    abst = [r for r in last if r.get("status") == "abstain"]
    print(f"last60 abstains: {len(abst)}\n")

    # 3分類
    buckets = defaultdict(list)
    for r in abst:
        bucket, note = classify_row(r)
        buckets[bucket].append((r, note))

    # サマリ
    cnt = {k: len(v) for k, v in buckets.items() if k != "-"}
    print("=== bucket counts ===")
    for k in ["A", "B", "C"]:
        print(f"{k}: {cnt.get(k, 0)}")
    print("")

    # 明細
    for k in ["A", "B", "C"]:
        print(f"--- bucket {k} ---")
        for r, note in buckets.get(k, []):
            q = r.get("query")
            used = r.get("collection_used")
            hit = r.get("tourism_dec_hit") or r.get("abstain_hit")
            dist = r.get("main_best_dist")
            th = r.get("dist_threshold_main")
            print(f"- {q} | used={used} | hit={hit} | dist={dist} th={th} | note={note}")
        print("")

    # 参考：abstain理由の内訳
    reasons = Counter((r.get("tourism_dec_reason") for r in abst if r.get("category") == "観光"))
    if reasons:
        print("=== tourism_dec_reason breakdown (abstain only) ===")
        for k, v in reasons.most_common():
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()