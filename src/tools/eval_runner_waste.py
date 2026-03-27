from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import chromadb
from chromadb.config import Settings
from openai import OpenAI

# =========================
# 設定（あなたの環境に合わせて調整）
# =========================
BASE_DIR = Path(__file__).resolve().parent  # src/
EVAL_CSV = BASE_DIR / "data" / "eval" / "eval_waste_50.csv"
OUT_CSV  = BASE_DIR / "data" / "eval" / "eval_waste_50_results.csv"

# Chroma 永続パス：
# app_portal_v2.py の WASTE_LIFE_DB_DIR と同じ場所に合わせてください。
# 例：src/chroma_db, src/db, appで使っている永続Dir etc.
CHROMA_DIR = BASE_DIR / "chroma_db"

# コレクション名（FAQ→本体）
COL_FAQ  = "okazaki_waste_faq"
COL_MAIN = "okazaki_waste"

# OpenAI
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4.1-mini"

# dist閾値（ごみ）
# ※あなたの現行スケール（0.7〜1.3）に合わせたデフォルト
DIST_THRESHOLD_WASTE = 1.10

TOP_K = 5

SYSTEM_PROMPT = """あなたは自治体の公式資料に基づいて回答する行政向けAIです。
- 根拠は提示された資料（コンテキスト）だけ。
- 根拠がない/弱い場合は必ず abstain する。
- 断定できない時は、断定せずに「資料の該当箇所の確認が必要」と述べる。
"""

# =========================
# ユーティリティ
# =========================
def normalize_jp(s: str) -> str:
    s = s.replace("　", " ")
    s = re.sub(r"\s+", "", s)
    return s

def extract_terms(s: str) -> List[str]:
    # expected_terms が空なら空配列
    s = (s or "").strip()
    if not s:
        return []
    return [t.strip() for t in s.split(",") if t.strip()]

def evidence_hits(text: str, terms: List[str]) -> Tuple[bool, List[str]]:
    if not terms:
        return True, []
    t = normalize_jp(text)
    hit = []
    for term in terms:
        if normalize_jp(term) in t:
            hit.append(term)
    return (len(hit) >= 1), hit

def build_context(res: Dict[str, Any], max_chars: int = 2200) -> Tuple[str, List[str]]:
    docs0 = res.get("documents", [[]])[0] if res.get("documents") else []
    metas0 = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
    ctx_parts = []
    cites = []
    used = 0
    for doc, meta in zip(docs0, metas0):
        if not doc:
            continue
        snippet = doc.strip()
        if not snippet:
            continue
        # 出典URL
        url = (meta or {}).get("pdf_url") or (meta or {}).get("file_url") or (meta or {}).get("url") or ""
        page = (meta or {}).get("page") or (meta or {}).get("page_start") or ""
        if url:
            cites.append(f"{url} p.{page}".strip())
        # コンテキストを詰める
        if used + len(snippet) > max_chars:
            snippet = snippet[: max(0, max_chars - used)]
        ctx_parts.append(snippet)
        used += len(snippet)
        if used >= max_chars:
            break
    return "\n\n---\n\n".join(ctx_parts), cites

def rerank_pdf_first(res: Dict[str, Any]) -> Dict[str, Any]:
    # 最小：PDF優先（イベント抑制はこの評価では不要）
    docs0 = res.get("documents", [[]])[0]
    metas0 = res.get("metadatas", [[]])[0]
    dists0 = res.get("distances", [[]])[0]

    scored = []
    for i, (m, d) in enumerate(zip(metas0, dists0)):
        url = ((m or {}).get("pdf_url") or (m or {}).get("file_url") or (m or {}).get("url") or "")
        adj = float(d)
        if url.lower().endswith(".pdf"):
            adj -= 0.05
        scored.append((adj, i))
    scored.sort(key=lambda x: x[0])

    order = [i for _, i in scored]
    adj_map = {i: adj for (adj, i) in scored}

    res["documents"] = [[docs0[i] for i in order]]
    res["metadatas"] = [[metas0[i] for i in order]]
    # ★重要：調整後距離を distances に入れる（あなたのアプリ修正方針と同じ）
    res["distances"] = [[float(adj_map[i]) for i in order]]
    return res

@dataclass
class AnswerResult:
    status: str           # ok / abstain
    collection: str       # faq or main
    best_dist: float
    answer: str
    cites: List[str]
    matched_terms: List[str]

def answer_one(
    oa: OpenAI,
    col,
    user_query: str,
    expected_terms: List[str],
    *,
    dist_threshold: float,
) -> AnswerResult:
    # embedding
    emb = oa.embeddings.create(model=EMBED_MODEL, input=[user_query]).data[0].embedding
    res = col.query(
        query_embeddings=[emb],
        n_results=TOP_K,
        where=None,
        include=["documents", "metadatas", "distances"],
    )
    res = rerank_pdf_first(res)
    best_dist = float(res.get("distances", [[999]])[0][0])

    ctx, cites = build_context(res)
    if best_dist > dist_threshold:
        return AnswerResult(
            status="abstain",
            collection="",
            best_dist=best_dist,
            answer="資料内で根拠を確認できなかったため、推測で回答できません。",
            cites=cites,
            matched_terms=[],
        )

    # evidence gate（期待語がある場合のみ）
    ok_ev, matched = evidence_hits(ctx, expected_terms)
    if not ok_ev:
        return AnswerResult(
            status="abstain",
            collection="",
            best_dist=best_dist,
            answer="資料内で質問に直接関係する記載（用語・言い換え語）を確認できなかったため、推測で回答できません。",
            cites=cites,
            matched_terms=matched,
        )

    user_prompt = f"""質問: {user_query}

参考資料（抜粋）:
{ctx}

上の資料だけを根拠に日本語で回答してください。
最後に必ず「根拠（URL / ページ）」を箇条書きで出してください。
"""
    chat = oa.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    ans = (chat.choices[0].message.content or "").strip()
    return AnswerResult(
        status="ok",
        collection="",
        best_dist=best_dist,
        answer=ans,
        cites=cites,
        matched_terms=matched,
    )

def main():
    # OpenAI key
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY が環境変数にありません。.env か PowerShell の $env:OPENAI_API_KEY を設定してください。")

    oa = OpenAI()

    # Chroma
    if not CHROMA_DIR.exists():
        raise RuntimeError(f"Chroma dir not found: {CHROMA_DIR}（appと同じ永続パスに合わせてください）")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(allow_reset=False))

    # コレクション取得（FAQは無ければスキップ）
    faq_col = None
    try:
        faq_col = client.get_collection(COL_FAQ)
    except Exception:
        faq_col = None

    main_col = client.get_collection(COL_MAIN)

    # eval読み込み
    rows = []
    with EVAL_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # 実行
    results = []
    cnt_should = 0
    cnt_ans_on_should = 0
    cnt_false_answer = 0

    for r in rows:
        qid = r["qid"].strip()
        q = r["user_question"].strip()
        should = int(r["should_answer"])
        terms = extract_terms(r.get("expected_terms", ""))

        # 1) FAQ優先
        used_col = "main"
        res1 = None

        if faq_col is not None:
            res1 = answer_one(oa, faq_col, q, terms, dist_threshold=DIST_THRESHOLD_WASTE)
            if res1.status == "ok":
                used_col = "faq"
                final = res1
            else:
                final = None
        else:
            final = None

        # 2) main fallback
        if final is None:
            res2 = answer_one(oa, main_col, q, terms, dist_threshold=DIST_THRESHOLD_WASTE)
            used_col = "main"
            final = res2

        # 集計
        if should == 1:
            cnt_should += 1
            if final.status == "ok":
                cnt_ans_on_should += 1
        else:
            if final.status == "ok":
                cnt_false_answer += 1

        # 出力用
        ans_snip = final.answer.replace("\n", " ")[:140]
        results.append({
            "qid": qid,
            "user_question": q,
            "should_answer": should,
            "status": final.status,
            "collection_used": used_col,
            "best_dist": f"{final.best_dist:.4f}",
            "cites_count": str(len(final.cites)),
            "matched_terms": ",".join(final.matched_terms),
            "answer_snippet": ans_snip,
        })
        print(f"[{qid}] {final.status} col={used_col} dist={final.best_dist:.3f} | {ans_snip}")

    # 書き出し
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    # サマリ
    answer_rate = (cnt_ans_on_should / cnt_should) if cnt_should else 0.0
    print("\n==== SUMMARY ====")
    print(f"Should-answer count: {cnt_should}")
    print(f"Answered on should: {cnt_ans_on_should}")
    print(f"Answer rate (on should): {answer_rate:.1%}")
    print(f"False-answer count (should_answer=0 but ok): {cnt_false_answer}")
    print(f"Results saved: {OUT_CSV}")

if __name__ == "__main__":
    main()
