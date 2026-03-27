#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_logs.py
---------------------------------
qa_log.jsonl（1行=1JSON）を解析して、改善に必要な集計CSVを出力します。
※ 単体で動く（依存なし：標準ライブラリのみ）

出力（outdir配下）:
  1) summary.json                全体サマリ
  2) by_category.csv             カテゴリ別集計（abstain率/FAQ率/距離統計など）
  3) by_answered_by.csv          answered_by別集計（main/faq/none）
  4) candidates_focus.csv        “改善優先”候補（near-threshold-abstain / no-cite-ok / slow）
  5) dist_bins_main.csv          main_best_dist のヒスト（カテゴリ別）
  6) dist_bins_faq.csv           faq_best_dist のヒスト（カテゴリ別）
  7) top_queries.csv             頻出query（正規化キーで集約）

想定ログキー（無くても動く）:
  ts, query, chosen_cat/category, status, answered_by,
  main_best_dist, faq_best_dist, dist_threshold_main, dist_threshold_faq,
  num_cites, latency_ms, collection_used, collection

使い方:
  python analyze_logs.py --log logs/qa_log.jsonl --out out --min-freq 2

おすすめ:
  - out/by_category.csv をExcelでピボットすると改善ポイントが一発で見えます。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ----------------------------
# 正規化（同一質問の集約用）
# ----------------------------
PUNCT_RE = re.compile(r"[　\s]+")
TRIM_RE = re.compile(r"^[\s　]+|[\s　]+$")

def normalize_query(q: str) -> str:
    if not q:
        return ""
    s = q.replace("\u3000", " ")
    s = TRIM_RE.sub("", s)
    s = re.sub(r"[、，,。\.！!？\?「」『』（）\(\)\[\]\{\}<>＜＞【】]", " ", s)
    s = PUNCT_RE.sub(" ", s)
    return s.lower()


def safe_get(d: Dict[str, Any], *keys: str, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def to_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            return int(x)
        s = str(x).strip()
        if s == "":
            return None
        return int(float(s))
    except Exception:
        return None


def to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            return float(x)
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue


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


def summarize_numbers(xs: List[Optional[float]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    vals = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    if not vals:
        return None, None, None
    return round(min(vals), 4), round(median(vals), 4), round(max(vals), 4)


def ratio(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return round(num / den, 4)


# ----------------------------
# dist binning（ヒスト）
# ----------------------------
def bin_edges(start: float, end: float, step: float) -> List[Tuple[float, float]]:
    bins = []
    x = start
    while x < end:
        bins.append((round(x, 3), round(x + step, 3)))
        x += step
    return bins


DEFAULT_BINS = bin_edges(0.0, 2.5, 0.1)  # distは運用により異なるが、まずは0-2.5で見る


def bin_value(v: Optional[float], bins: List[Tuple[float, float]]) -> Optional[str]:
    if v is None:
        return None
    for lo, hi in bins:
        if lo <= v < hi:
            return f"{lo:.1f}–{hi:.1f}"
    if v >= bins[-1][1]:
        return f"{bins[-1][1]:.1f}+"
    return None


# ----------------------------
# 解析本体
# ----------------------------
def analyze(log_path: str, outdir: str, min_freq: int = 2) -> None:
    rows = list(read_jsonl(log_path))
    if not rows:
        raise SystemExit("No valid JSONL rows found.")

    # 基本抽出
    parsed = []
    for o in rows:
        q = str(safe_get(o, "query", default="") or "")
        cat = str(safe_get(o, "chosen_cat", "category", default="") or "")
        status = str(safe_get(o, "status", default="") or "")
        answered_by = str(safe_get(o, "answered_by", default="none") or "none")
        latency_ms = to_int(safe_get(o, "latency_ms"))
        num_cites = to_int(safe_get(o, "num_cites")) or 0

        main_best = to_float(safe_get(o, "main_best_dist"))
        faq_best = to_float(safe_get(o, "faq_best_dist"))
        th_main = to_float(safe_get(o, "dist_threshold_main"))
        th_faq = to_float(safe_get(o, "dist_threshold_faq"))

        parsed.append({
            "ts": safe_get(o, "ts", default=""),
            "query": q,
            "norm_query": normalize_query(q),
            "category": cat if cat else "unknown",
            "status": status if status else "unknown",
            "answered_by": answered_by if answered_by else "none",
            "latency_ms": latency_ms,
            "num_cites": num_cites,
            "main_best_dist": main_best,
            "faq_best_dist": faq_best,
            "dist_threshold_main": th_main,
            "dist_threshold_faq": th_faq,
            "collection": safe_get(o, "collection", default=""),
            "collection_used": safe_get(o, "collection_used", default=""),
        })

    total = len(parsed)
    status_cnt = Counter(r["status"] for r in parsed)
    answered_cnt = Counter(r["answered_by"] for r in parsed)

    # 全体サマリ
    overall = {
        "total": total,
        "status_ok": status_cnt.get("ok", 0),
        "status_abstain": status_cnt.get("abstain", 0),
        "status_error": status_cnt.get("error", 0),
        "status_unknown": status_cnt.get("unknown", 0),
        "abstain_rate": ratio(status_cnt.get("abstain", 0), total),
        "answered_by_main": answered_cnt.get("main", 0),
        "answered_by_faq": answered_cnt.get("faq", 0),
        "answered_by_none": answered_cnt.get("none", 0),
    }

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    # カテゴリ別集計
    by_cat = defaultdict(list)
    for r in parsed:
        by_cat[r["category"]].append(r)

    by_category_rows: List[Dict[str, Any]] = []
    for cat, rs in sorted(by_cat.items(), key=lambda x: len(x[1]), reverse=True):
        n = len(rs)
        sc = Counter(x["status"] for x in rs)
        ac = Counter(x["answered_by"] for x in rs)

        main_min, main_med, main_max = summarize_numbers([x["main_best_dist"] for x in rs])
        faq_min, faq_med, faq_max = summarize_numbers([x["faq_best_dist"] for x in rs])

        lat_min, lat_med, lat_max = summarize_numbers([float(x["latency_ms"]) for x in rs if x["latency_ms"] is not None])

        # 閾値はカテゴリ内で同じ想定だが、念のため代表値
        th_main_vals = [x["dist_threshold_main"] for x in rs if x["dist_threshold_main"] is not None]
        th_faq_vals = [x["dist_threshold_faq"] for x in rs if x["dist_threshold_faq"] is not None]
        th_main_rep = round(median(th_main_vals), 4) if th_main_vals else None
        th_faq_rep = round(median(th_faq_vals), 4) if th_faq_vals else None

        # near-threshold-abstain: abstain & main_best <= th_main+0.05 & cites>0
        near = 0
        for x in rs:
            if x["status"] != "abstain":
                continue
            if x["main_best_dist"] is None or x["dist_threshold_main"] is None:
                continue
            if x["num_cites"] <= 0:
                continue
            if x["main_best_dist"] <= x["dist_threshold_main"] + 0.05:
                near += 1

        # no-cite-ok: okなのに citesが0 → 根拠抽出の穴疑い
        no_cite_ok = sum(1 for x in rs if x["status"] == "ok" and x["num_cites"] <= 0)

        by_category_rows.append({
            "category": cat,
            "n": n,
            "ok": sc.get("ok", 0),
            "abstain": sc.get("abstain", 0),
            "error": sc.get("error", 0),
            "abstain_rate": ratio(sc.get("abstain", 0), n),

            "answered_by_main": ac.get("main", 0),
            "answered_by_faq": ac.get("faq", 0),
            "answered_by_none": ac.get("none", 0),
            "faq_share": ratio(ac.get("faq", 0), n),

            "main_dist_min": main_min,
            "main_dist_median": main_med,
            "main_dist_max": main_max,

            "faq_dist_min": faq_min,
            "faq_dist_median": faq_med,
            "faq_dist_max": faq_max,

            "dist_threshold_main": th_main_rep,
            "dist_threshold_faq": th_faq_rep,

            "near_threshold_abstain": near,
            "near_threshold_abstain_rate": ratio(near, n),

            "no_cite_ok": no_cite_ok,
            "no_cite_ok_rate": ratio(no_cite_ok, n),

            "latency_ms_min": lat_min,
            "latency_ms_median": lat_med,
            "latency_ms_max": lat_max,
        })

    write_csv(os.path.join(outdir, "by_category.csv"), by_category_rows)

    # answered_by別集計
    by_ab = defaultdict(list)
    for r in parsed:
        by_ab[r["answered_by"]].append(r)

    by_answered_rows: List[Dict[str, Any]] = []
    for ab, rs in sorted(by_ab.items(), key=lambda x: len(x[1]), reverse=True):
        n = len(rs)
        sc = Counter(x["status"] for x in rs)
        main_min, main_med, main_max = summarize_numbers([x["main_best_dist"] for x in rs])
        faq_min, faq_med, faq_max = summarize_numbers([x["faq_best_dist"] for x in rs])
        by_answered_rows.append({
            "answered_by": ab,
            "n": n,
            "ok": sc.get("ok", 0),
            "abstain": sc.get("abstain", 0),
            "error": sc.get("error", 0),
            "abstain_rate": ratio(sc.get("abstain", 0), n),
            "main_dist_median": main_med,
            "faq_dist_median": faq_med,
        })

    write_csv(os.path.join(outdir, "by_answered_by.csv"), by_answered_rows)

    # 改善フォーカス候補（行単位）
    focus = []
    for r in parsed:
        # 1) near-threshold-abstain（惜しい）
        near = (
            r["status"] == "abstain"
            and r["main_best_dist"] is not None
            and r["dist_threshold_main"] is not None
            and r["num_cites"] > 0
            and r["main_best_dist"] <= r["dist_threshold_main"] + 0.05
        )
        # 2) no-cite-ok（根拠抽出の穴）
        no_cite_ok = (r["status"] == "ok" and r["num_cites"] <= 0)
        # 3) slow（遅い）
        slow = (r["latency_ms"] is not None and r["latency_ms"] >= 3000)

        if near or no_cite_ok or slow:
            focus.append({
                "ts": r["ts"],
                "category": r["category"],
                "status": r["status"],
                "answered_by": r["answered_by"],
                "main_best_dist": r["main_best_dist"],
                "faq_best_dist": r["faq_best_dist"],
                "dist_threshold_main": r["dist_threshold_main"],
                "dist_threshold_faq": r["dist_threshold_faq"],
                "num_cites": r["num_cites"],
                "latency_ms": r["latency_ms"],
                "flag_near_threshold_abstain": int(near),
                "flag_no_cite_ok": int(no_cite_ok),
                "flag_slow": int(slow),
                "query": r["query"],
            })

    write_csv(os.path.join(outdir, "candidates_focus.csv"), focus)

    # distヒスト（カテゴリ別）
    def dist_bins(rows_: List[Dict[str, Any]], key: str, out_name: str):
        # cat -> bin -> count
        m = defaultdict(Counter)
        for r in rows_:
            cat = r["category"]
            b = bin_value(r.get(key), DEFAULT_BINS)
            if b is None:
                continue
            m[cat][b] += 1

        # 出力を縦持ち（Excelピボット向け）
        out = []
        for cat, bc in m.items():
            for b, cnt in bc.items():
                out.append({"category": cat, "bin": b, "count": cnt})
        # count desc
        out.sort(key=lambda x: (x["category"], x["bin"]))
        write_csv(os.path.join(outdir, out_name), out)

    dist_bins(parsed, "main_best_dist", "dist_bins_main.csv")
    dist_bins(parsed, "faq_best_dist", "dist_bins_faq.csv")

    # 頻出query（正規化キーで集約）
    groups = defaultdict(list)
    for r in parsed:
        if r["norm_query"]:
            groups[r["norm_query"]].append(r)

    top_queries = []
    for norm_q, rs in groups.items():
        if len(rs) < min_freq:
            continue
        cat = Counter(x["category"] for x in rs).most_common(1)[0][0]
        status = Counter(x["status"] for x in rs)
        ab = Counter(x["answered_by"] for x in rs)
        sample = next((x["query"] for x in rs if x["query"]), "")
        top_queries.append({
            "norm_query": norm_q,
            "freq": len(rs),
            "category_major": cat,
            "ok": status.get("ok", 0),
            "abstain": status.get("abstain", 0),
            "abstain_rate": ratio(status.get("abstain", 0), len(rs)),
            "answered_by_faq": ab.get("faq", 0),
            "answered_by_main": ab.get("main", 0),
            "example_query": sample,
        })

    top_queries.sort(key=lambda x: x["freq"], reverse=True)
    write_csv(os.path.join(outdir, "top_queries.csv"), top_queries)

    print(f"OK: analyzed {total} rows")
    print(f" -> {os.path.join(outdir, 'summary.json')}")
    print(f" -> {os.path.join(outdir, 'by_category.csv')}")
    print(f" -> {os.path.join(outdir, 'candidates_focus.csv')}")
    print(f" -> {os.path.join(outdir, 'top_queries.csv')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="path to qa_log.jsonl")
    ap.add_argument("--out", default="./out", help="output directory")
    ap.add_argument("--min-freq", type=int, default=2, help="min frequency for top_queries.csv")
    args = ap.parse_args()
    analyze(args.log, args.out, min_freq=args.min_freq)


if __name__ == "__main__":
    main()
