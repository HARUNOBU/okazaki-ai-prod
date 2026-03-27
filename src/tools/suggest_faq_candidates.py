#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
suggest_faq_candidates.py
---------------------------------
qa_log.jsonl（1行=1JSON）から、FAQ化候補を自動抽出してCSV出力します。

想定ログキー（無くても動くように頑丈にしてあります）:
  ts, query, chosen_cat / category, status, answered_by,
  main_best_dist, faq_best_dist, dist_threshold_main, dist_threshold_faq,
  num_cites, cites, top_sources, year, mode, forced_cat, collection, collection_used

出力:
  - candidates.csv（候補の集約）
  - candidates_examples.csv（候補ごとの例クエリ）
使い方:
  python suggest_faq_candidates.py --log ./qa_log.jsonl --out ./out --min-freq 3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ----------------------------
# 設定（必要ならここだけ触る）
# ----------------------------
DANGER_PATTERNS = [
    r"どこで買", r"買える", r"販売", r"売って", r"購入", r"取扱", r"取り扱い", r"入手",
    r"おすすめ", r"オススメ", r"ベスト", r"最適", r"安い", r"比較", r"ランキング",
]
DANGER_RE = re.compile("|".join(DANGER_PATTERNS))

# 「正規化」で削りすぎない：年・日付・型番など数字は残す
PUNCT_RE = re.compile(r"[　\s]+")
TRIM_RE = re.compile(r"^[\s　]+|[\s　]+$")


def safe_get(d: Dict[str, Any], *keys: str, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def normalize_query(q: str) -> str:
    """FAQ候補抽出用のゆるい正規化（意味を壊しにくい）"""
    if not q:
        return ""
    s = q
    s = s.replace("\u3000", " ")  # 全角スペース
    s = TRIM_RE.sub("", s)
    # 句読点類は落とす（ただし数字は残す）
    s = re.sub(r"[、，,。\.！!？\?「」『』（）\(\)\[\]\{\}<>＜＞【】]", " ", s)
    # 連続空白を1つに
    s = PUNCT_RE.sub(" ", s)
    return s.lower()


def is_danger_query(q: str) -> bool:
    return bool(DANGER_RE.search(q or ""))


def extract_top_sources(obj: Dict[str, Any], limit: int = 3) -> List[str]:
    """
    ログの top_sources / cites から、"url#page" っぽい代表ソースを最大limit件抽出。
    """
    out: List[str] = []

    # 1) top_sources があれば優先（推奨）
    ts = obj.get("top_sources")
    if isinstance(ts, list):
        for it in ts:
            if len(out) >= limit:
                break
            if isinstance(it, dict):
                url = (it.get("url") or "").strip()
                page = it.get("page")
                if url:
                    if isinstance(page, int):
                        out.append(f"{url}#p{page}")
                    else:
                        out.append(url)
            elif isinstance(it, str):
                out.append(it[:200])

    if len(out) >= limit:
        return out[:limit]

    # 2) cites から雑に拾う
    cites = obj.get("cites")
    if isinstance(cites, list):
        for c in cites:
            if len(out) >= limit:
                break
            if not isinstance(c, str):
                continue
            s = c.strip()
            url = ""
            for tok in s.split():
                if tok.startswith("http://") or tok.startswith("https://"):
                    url = tok
                    break
            if url and url not in out:
                out.append(url)

    return out[:limit]


@dataclass
class LogRow:
    ts: str
    query: str
    norm_query: str
    category: str
    status: str
    answered_by: str
    main_best_dist: Optional[float]
    faq_best_dist: Optional[float]
    dist_threshold_main: Optional[float]
    dist_threshold_faq: Optional[float]
    num_cites: int
    sources: List[str]
    raw: Dict[str, Any]


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                # 壊れた行があっても止めない
                continue


def parse_rows(path: str) -> List[LogRow]:
    rows: List[LogRow] = []
    for obj in read_jsonl(path):
        q = safe_get(obj, "query", default="")
        cat = safe_get(obj, "chosen_cat", "category", default="")
        status = safe_get(obj, "status", default="")
        answered_by = safe_get(obj, "answered_by", default="none")
        m = to_float(safe_get(obj, "main_best_dist"))
        f = to_float(safe_get(obj, "faq_best_dist"))
        th_m = to_float(safe_get(obj, "dist_threshold_main"))
        th_f = to_float(safe_get(obj, "dist_threshold_faq"))
        num_cites = safe_get(obj, "num_cites", default=0)
        try:
            num_cites = int(num_cites) if num_cites is not None else 0
        except Exception:
            num_cites = 0

        sources = extract_top_sources(obj, limit=3)
        ts = safe_get(obj, "ts", default="")

        rows.append(
            LogRow(
                ts=str(ts),
                query=str(q),
                norm_query=normalize_query(str(q)),
                category=str(cat),
                status=str(status),
                answered_by=str(answered_by),
                main_best_dist=m,
                faq_best_dist=f,
                dist_threshold_main=th_m,
                dist_threshold_faq=th_f,
                num_cites=num_cites,
                sources=sources,
                raw=obj,
            )
        )
    return rows


def score_candidate(
    freq: int,
    status_counts: Counter,
    answered_by_counts: Counter,
    danger_ratio: float,
    abstain_ratio: float,
    near_threshold_ratio: float,
    faq_proven_ratio: float,
    no_cite_ratio: float,
) -> float:
    # 保守的（幻覚ゼロ志向）：頻度と「危険/近いのに答えられない」を強く評価
    s = 0.0
    s += min(freq, 10) * 1.0

    # abstainが多い＝「困ってる」ので上げる（ただし根拠ゼロばかりなら別扱い）
    s += abstain_ratio * 5.0

    # 閾値付近でabstain（＝惜しい）を強く上げる
    s += near_threshold_ratio * 7.0

    # FAQで実際に救済できているなら価値が実証済み
    s += faq_proven_ratio * 4.0

    # 危険質問（買える/おすすめ等）は「答えないFAQ」含め価値が高い
    s += danger_ratio * 8.0

    # citesがゼロが多いのにokが多いのは怪しい（根拠抽出の穴の可能性）
    s -= no_cite_ratio * 2.0

    return round(s, 3)


def classify_candidate(
    freq: int,
    abstain_ratio: float,
    near_threshold_ratio: float,
    faq_proven_ratio: float,
    danger_ratio: float,
) -> str:
    # Type分類（複数当てはまる場合は優先順）
    if danger_ratio >= 0.4:
        return "Type-4: danger(abstain-template)"
    if near_threshold_ratio >= 0.25:
        return "Type-2: near-threshold-abstain"
    if faq_proven_ratio >= 0.25 and freq >= 2:
        return "Type-3: faq-proven"
    if freq >= 3:
        return "Type-1: frequent"
    # それ以外は低優先
    return "Type-0: low"


def summarize_dist(vals: List[Optional[float]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    xs = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    if not xs:
        return (None, None, None)
    return (round(min(xs), 4), round(median(xs), 4), round(max(xs), 4))


def build_candidates(rows: List[LogRow], min_freq: int = 3) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    groups: Dict[str, List[LogRow]] = defaultdict(list)
    for r in rows:
        if not r.norm_query:
            continue
        groups[r.norm_query].append(r)

    candidates: List[Dict[str, Any]] = []
    examples: List[Dict[str, Any]] = []

    for norm_q, rs in groups.items():
        freq = len(rs)
        if freq < min_freq:
            continue

        status_counts = Counter(r.status for r in rs)
        answered_by_counts = Counter(r.answered_by for r in rs)

        danger_cnt = sum(1 for r in rs if is_danger_query(r.query))
        abstain_cnt = status_counts.get("abstain", 0)

        # near-threshold-abstain: abstain かつ main_best_dist <= th_main + 0.05 かつ cites>0
        near_cnt = 0
        for r in rs:
            if r.status != "abstain":
                continue
            if r.main_best_dist is None or r.dist_threshold_main is None:
                continue
            if r.num_cites <= 0:
                continue
            if r.main_best_dist <= (r.dist_threshold_main + 0.05):
                near_cnt += 1

        faq_proven_cnt = 0
        for r in rs:
            if r.answered_by != "faq":
                continue
            # faq_best_dist <= th_faq が取れるなら条件に使う
            if r.faq_best_dist is not None and r.dist_threshold_faq is not None:
                if r.faq_best_dist <= r.dist_threshold_faq:
                    faq_proven_cnt += 1
            else:
                faq_proven_cnt += 1  # 閾値情報が無ければ一旦カウント

        no_cite_cnt = sum(1 for r in rs if r.num_cites <= 0)
        danger_ratio = danger_cnt / freq
        abstain_ratio = abstain_cnt / freq
        near_ratio = near_cnt / freq
        faq_proven_ratio = faq_proven_cnt / freq
        no_cite_ratio = no_cite_cnt / freq

        cand_type = classify_candidate(freq, abstain_ratio, near_ratio, faq_proven_ratio, danger_ratio)

        # 代表カテゴリ：最頻
        cat = Counter(r.category for r in rs).most_common(1)[0][0]

        # dist統計
        main_min, main_med, main_max = summarize_dist([r.main_best_dist for r in rs])
        faq_min, faq_med, faq_max = summarize_dist([r.faq_best_dist for r in rs])

        # 代表ソース（頻度順）
        src_counter = Counter()
        for r in rs:
            for s in r.sources:
                src_counter[s] += 1
        top_sources = [s for s, _ in src_counter.most_common(3)]

        # 推奨アクション
        if cand_type.startswith("Type-4"):
            action = "Create FAQ (abstain-template). Provide official reference/route, do NOT guess."
        elif cand_type.startswith(("Type-1", "Type-3", "Type-2")):
            action = "Create FAQ (answer). Fix wording, pin citations (pdf_url/page), add aliases."
        else:
            action = "Investigate. Likely needs data/citation extraction improvements."

        score = score_candidate(
            freq=freq,
            status_counts=status_counts,
            answered_by_counts=answered_by_counts,
            danger_ratio=danger_ratio,
            abstain_ratio=abstain_ratio,
            near_threshold_ratio=near_ratio,
            faq_proven_ratio=faq_proven_ratio,
            no_cite_ratio=no_cite_ratio,
        )

        # 例クエリ（最大3）
        example_qs = []
        for r in sorted(rs, key=lambda x: x.ts or ""):
            if r.query and r.query not in example_qs:
                example_qs.append(r.query)
            if len(example_qs) >= 3:
                break

        candidates.append({
            "score": score,
            "candidate_type": cand_type,
            "category": cat,
            "norm_query": norm_q,
            "freq": freq,
            "status_ok": status_counts.get("ok", 0),
            "status_abstain": status_counts.get("abstain", 0),
            "status_error": status_counts.get("error", 0),
            "answered_by_faq": answered_by_counts.get("faq", 0),
            "answered_by_main": answered_by_counts.get("main", 0),
            "danger_ratio": round(danger_ratio, 3),
            "abstain_ratio": round(abstain_ratio, 3),
            "near_threshold_ratio": round(near_ratio, 3),
            "faq_proven_ratio": round(faq_proven_ratio, 3),
            "no_cite_ratio": round(no_cite_ratio, 3),
            "main_dist_min": main_min,
            "main_dist_median": main_med,
            "main_dist_max": main_max,
            "faq_dist_min": faq_min,
            "faq_dist_median": faq_med,
            "faq_dist_max": faq_max,
            "top_sources": " | ".join(top_sources),
            "suggested_action": action,
            "example_q1": example_qs[0] if len(example_qs) > 0 else "",
            "example_q2": example_qs[1] if len(example_qs) > 1 else "",
            "example_q3": example_qs[2] if len(example_qs) > 2 else "",
        })

        # examples: 生ログ参照用（候補ごとに最大10件）
        for r in rs[:10]:
            examples.append({
                "norm_query": norm_q,
                "ts": r.ts,
                "category": r.category,
                "status": r.status,
                "answered_by": r.answered_by,
                "main_best_dist": r.main_best_dist,
                "faq_best_dist": r.faq_best_dist,
                "dist_threshold_main": r.dist_threshold_main,
                "dist_threshold_faq": r.dist_threshold_faq,
                "num_cites": r.num_cites,
                "query": r.query,
                "sources": " | ".join(r.sources),
            })

    # スコア順に並べる
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates, examples


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="path to qa_log.jsonl")
    ap.add_argument("--out", default="./out", help="output directory")
    ap.add_argument("--min-freq", type=int, default=3, help="minimum frequency to consider")
    ap.add_argument("--top", type=int, default=50, help="keep top N candidates by score")
    args = ap.parse_args()

    rows = parse_rows(args.log)
    if not rows:
        print("No valid rows found. Check your --log path or JSONL format.")
        return

    candidates, examples = build_candidates(rows, min_freq=args.min_freq)

    # 上位だけ残す（βでは多すぎると回らない）
    candidates = candidates[: max(1, args.top)]

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    write_csv(os.path.join(out_dir, "candidates.csv"), candidates)
    write_csv(os.path.join(out_dir, "candidates_examples.csv"), examples)

    print(f"OK: wrote {len(candidates)} candidates -> {os.path.join(out_dir, 'candidates.csv')}")
    print(f"OK: wrote {len(examples)} example rows -> {os.path.join(out_dir, 'candidates_examples.csv')}")
    print("Next: open candidates.csv, start with top 5 only (avoid FAQ bloat).")


if __name__ == "__main__":
    main()
