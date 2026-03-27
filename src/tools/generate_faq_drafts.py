#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_faq_drafts.py
---------------------------------
qa_log.jsonl から「FAQカード雛形」を半自動生成します。

目的：
- 候補（norm_query単位）を集約し、FAQカードの“器”を作る
- Type-4（危険質問）は「答えないFAQ（abstainテンプレ）」を自動で下書き
- それ以外は「回答本文は空（人が埋める）」で安全に運用（幻覚ゼロ）

出力：
- faq_drafts.json  : build_waste_faq.py の upsert で使える形に近い汎用JSON（question/answer/aliases/meta）
- faq_drafts.csv   : Excel編集用

使い方：
  python generate_faq_drafts.py --log ./qa_log.jsonl --out ./out --min-freq 2 --max 50

前提：
- qa_log.jsonl のキーが足りなくても動きます（頑丈に実装）
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# 設定（必要ならここだけ触る）
# ----------------------------
DANGER_PATTERNS = [
    r"どこで買", r"買える", r"販売", r"売って", r"購入", r"取扱", r"取り扱い", r"入手",
    r"おすすめ", r"オススメ", r"ベスト", r"最適", r"安い", r"比較", r"ランキング",
]
DANGER_RE = re.compile("|".join(DANGER_PATTERNS))

# alias 抽出時に落とす語（ノイズ抑制：必要なら足す）
STOPWORDS = set([
    "です", "ます", "したい", "について", "方法", "教えて", "ください", "何", "どこ", "いつ",
    "可能", "できます", "お願い", "岡崎", "岡崎市", "教えてください"
])

PUNCT_RE = re.compile(r"[　\s]+")
TRIM_RE = re.compile(r"^[\s　]+|[\s　]+$")


def safe_get(d: Dict[str, Any], *keys: str, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue


def normalize_query(q: str) -> str:
    """候補集約用のゆる正規化（意味を壊しにくい）"""
    if not q:
        return ""
    s = q.replace("\u3000", " ")
    s = TRIM_RE.sub("", s)
    s = re.sub(r"[、，,。\.！!？\?「」『』（）\(\)\[\]\{\}<>＜＞【】]", " ", s)
    s = PUNCT_RE.sub(" ", s)
    return s.lower()


def is_danger(q: str) -> bool:
    return bool(DANGER_RE.search(q or ""))


def extract_url_and_page_from_cite(s: str) -> Tuple[str, Optional[int]]:
    """cites/top_sourcesのraw文字列から雑に URL と page を拾う（拾えないなら空）"""
    url = ""
    page = None
    if not s:
        return url, page

    # URL
    for tok in s.split():
        if tok.startswith("http://") or tok.startswith("https://"):
            url = tok
            break

    # pageっぽい数字（最後の数値）
    digits = "".join(ch if ch.isdigit() else " " for ch in s).split()
    if digits:
        try:
            page = int(digits[-1])
        except Exception:
            page = None
    return url, page


def extract_sources(obj: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
    """
    ログから根拠候補を構造化して最大limit件返す。
    """
    out: List[Dict[str, Any]] = []

    ts = obj.get("top_sources")
    if isinstance(ts, list):
        for it in ts:
            if len(out) >= limit:
                break
            if isinstance(it, dict):
                url = (it.get("url") or "").strip()
                page = it.get("page")
                if url:
                    out.append({"url": url, "page": page if isinstance(page, int) else None, "raw": (it.get("raw") or "")[:200]})
            elif isinstance(it, str):
                url, page = extract_url_and_page_from_cite(it)
                out.append({"url": url, "page": page, "raw": it[:200]})

    if len(out) >= limit:
        return out[:limit]

    cites = obj.get("cites")
    if isinstance(cites, list):
        for c in cites:
            if len(out) >= limit:
                break
            if not isinstance(c, str):
                continue
            url, page = extract_url_and_page_from_cite(c)
            out.append({"url": url, "page": page, "raw": c[:200]})

    # urlが空のものを後ろへ
    out.sort(key=lambda x: (0 if x.get("url") else 1))
    # 重複除去（url#page）
    seen = set()
    uniq = []
    for x in out:
        key = f'{x.get("url","")}#{x.get("page")}'
        if key in seen:
            continue
        seen.add(key)
        uniq.append(x)
    return uniq[:limit]


def tokenize_for_alias(q: str) -> List[str]:
    """
    超簡易 alias 候補生成：
    - 記号除去
    - 2文字以上のトークン
    - stopwords除去
    ※ 日本語形態素は使わず“雑”にやる（安全・依存なし）
    """
    s = q
    s = s.replace("\u3000", " ")
    s = re.sub(r"[、，,。\.！!？\?「」『』（）\(\)\[\]\{\}<>＜＞【】]", " ", s)
    s = PUNCT_RE.sub(" ", s).strip()

    toks = []
    for t in s.split():
        t = t.strip()
        if len(t) < 2:
            continue
        if t in STOPWORDS:
            continue
        toks.append(t)

    # 同義語っぽく使えそうな“部分文字列”も少し作る（過剰増殖を防ぐ）
    # 例： "指定ごみ袋" -> "ごみ袋"
    extra = []
    for t in toks:
        if "指定" in t and len(t) >= 4:
            extra.append(t.replace("指定", ""))
    return list(dict.fromkeys(toks + extra))[:10]


def stable_id(prefix: str, norm_query: str) -> str:
    h = hashlib.sha1(norm_query.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}"


def make_abstain_template(category: str) -> str:
    """
    Type-4（危険質問）用の “答えないFAQ” テンプレ。
    根拠が資料内にある場合のみ追記して使う。
    """
    return (
        "資料内で根拠を確認できなかったため、推測で回答できません。\n\n"
        "お手数ですが、岡崎市の公式資料（ごみ出しガイドブック等）または公式窓口でご確認ください。\n"
        "※当アプリは資料に基づく範囲のみ回答します。"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="path to qa_log.jsonl")
    ap.add_argument("--out", default="./out", help="output directory")
    ap.add_argument("--min-freq", type=int, default=2, help="minimum frequency per normalized query")
    ap.add_argument("--max", type=int, default=50, help="max number of drafts")
    ap.add_argument("--prefix", default="faq", help="id prefix (e.g., waste_faq)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1) グルーピング
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for obj in read_jsonl(args.log):
        q = safe_get(obj, "query", default="")
        norm = normalize_query(str(q))
        if not norm:
            continue
        groups[norm].append(obj)

    # 2) 候補抽出（頻度 >= min_freq）
    norms = [n for n, rs in groups.items() if len(rs) >= args.min_freq]
    if not norms:
        print("No groups found with min_freq. Lower --min-freq or check log.")
        return

    # 3) 簡易スコア：頻度 + 危険 + abstain比率 + near-threshold-abstain(あれば)
    scored: List[Tuple[float, str]] = []
    for norm in norms:
        rs = groups[norm]
        freq = len(rs)

        status_counts = Counter(str(safe_get(r, "status", default="")) for r in rs)
        answered_by_counts = Counter(str(safe_get(r, "answered_by", default="none")) for r in rs)
        danger_cnt = sum(1 for r in rs if is_danger(str(safe_get(r, "query", default=""))))
        abstain_cnt = status_counts.get("abstain", 0)

        # near-threshold-abstain の雑検出（ログが入ってれば効く）
        near_cnt = 0
        for r in rs:
            if str(safe_get(r, "status", default="")) != "abstain":
                continue
            mbd = safe_get(r, "main_best_dist")
            thm = safe_get(r, "dist_threshold_main")
            num_cites = safe_get(r, "num_cites", default=0)
            try:
                mbd = float(mbd) if mbd is not None else None
                thm = float(thm) if thm is not None else None
                num_cites = int(num_cites) if num_cites is not None else 0
            except Exception:
                mbd, thm, num_cites = None, None, 0
            if mbd is not None and thm is not None and num_cites > 0 and mbd <= thm + 0.05:
                near_cnt += 1

        # スコア（保守的）
        danger_ratio = danger_cnt / freq
        abstain_ratio = abstain_cnt / freq
        near_ratio = near_cnt / freq
        faq_ratio = answered_by_counts.get("faq", 0) / freq

        score = 0.0
        score += min(freq, 10) * 1.0
        score += danger_ratio * 8.0
        score += abstain_ratio * 4.0
        score += near_ratio * 7.0
        score += faq_ratio * 3.0
        scored.append((round(score, 3), norm))

    scored.sort(key=lambda x: x[0], reverse=True)
    scored = scored[: max(1, args.max)]

    # 4) FAQドラフト生成
    drafts: List[Dict[str, Any]] = []
    drafts_csv: List[Dict[str, Any]] = []

    for score, norm in scored:
        rs = groups[norm]
        # 代表クエリ：最頻の生クエリ
        q_counter = Counter(str(safe_get(r, "query", default="")) for r in rs)
        rep_q = q_counter.most_common(1)[0][0]

        # 代表カテゴリ
        cat_counter = Counter(str(safe_get(r, "chosen_cat", safe_get(r, "category", default=""))) for r in rs)
        category = cat_counter.most_common(1)[0][0] or "unknown"

        # 危険判定
        danger_ratio = sum(1 for r in rs if is_danger(str(safe_get(r, "query", default="")))) / len(rs)
        is_danger_group = danger_ratio >= 0.4

        # alias候補：代表クエリ＋上位クエリから抽出
        aliases: List[str] = []
        for q, _ in q_counter.most_common(5):
            aliases.extend(tokenize_for_alias(q))
        # 重複除去
        aliases = list(dict.fromkeys([a for a in aliases if a]))[:15]

        # 根拠候補（上位3件）
        src_counter = Counter()
        src_samples: Dict[str, Dict[str, Any]] = {}
        for r in rs:
            for s in extract_sources(r, limit=3):
                key = f'{s.get("url","")}#{s.get("page")}'
                src_counter[key] += 1
                if key not in src_samples:
                    src_samples[key] = s
        top_src_keys = [k for k, _ in src_counter.most_common(3)]
        top_sources = [src_samples[k] for k in top_src_keys]

        # metaの代表：最頻url/page（あれば）
        pdf_url = ""
        page = None
        if top_sources:
            pdf_url = (top_sources[0].get("url") or "").strip()
            page = top_sources[0].get("page")

        # 回答本文（危険グループはabstainテンプレ、それ以外は空で人が埋める）
        answer_draft = make_abstain_template(category) if is_danger_group else ""

        # ID（norm_queryから安定生成）
        faq_id = stable_id(args.prefix, norm)

        # ドラフト（build_waste_faq.py 側の形に寄せる：id/text/meta でも使えるように）
        draft = {
            "id": faq_id,
            "question": rep_q,
            "answer": answer_draft,     # ←危険は自動、通常は空（人が埋める）
            "aliases": aliases,         # ←検索ヒット率UP用
            "meta": {
                "category": category,
                "pdf_url": pdf_url,
                "page": page,
                "top_sources": top_sources,  # 参考（後で削っても良い）
                "score": score,
                "freq": len(rs),
                "danger_group": is_danger_group,
            },
        }
        drafts.append(draft)

        drafts_csv.append({
            "id": faq_id,
            "category": category,
            "freq": len(rs),
            "score": score,
            "danger_group": is_danger_group,
            "question": rep_q,
            "answer_draft": answer_draft,
            "aliases": " | ".join(aliases),
            "pdf_url": pdf_url,
            "page": page if page is not None else "",
            "top_sources": " | ".join([f'{x.get("url","")}#p{x.get("page")}' for x in top_sources]),
        })

    # 5) 出力
    out_json = os.path.join(args.out, "faq_drafts.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(drafts, f, ensure_ascii=False, indent=2)

    out_csv = os.path.join(args.out, "faq_drafts.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(drafts_csv[0].keys()))
        w.writeheader()
        for r in drafts_csv:
            w.writerow(r)

    print(f"OK: wrote {len(drafts)} drafts -> {out_json}")
    print(f"OK: wrote {len(drafts_csv)} drafts -> {out_csv}")
    print("Next recommended workflow:")
    print("  1) Open faq_drafts.csv in Excel and fill 'answer_draft' for non-danger rows.")
    print("  2) For each filled row, keep citations consistent (pdf_url/page).")
    print("  3) Convert to your build_waste_faq.py input format (or adapt build script to read faq_drafts.json).")


if __name__ == "__main__":
    main()
