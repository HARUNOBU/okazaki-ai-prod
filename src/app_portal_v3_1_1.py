import json
from datetime import datetime

import time
import uuid
from datetime import datetime, timezone


from pathlib import Path
import os
import re
from typing import Dict, Any, List, Tuple

import streamlit as st
from abstain.tourism import decide_tourism_abstain, TOURISM_ABSTAIN_MESSAGE
from dotenv import load_dotenv
from config import (
    APP_ENV,
    DEBUG,
    OPENAI_API_KEY,
    WASTE_LIFE_DB_DIR,
    KANKO_DB_DIR,
    LOG_DIR,
    DATA_DIR,
    STREAMLIT_PORT,
)
from openai import OpenAI
import chromadb
from chromadb.config import Settings



# ===== DB設定 =====
#WASTE_LIFE_DB_DIR = "chroma_db"  # okazaki_waste_rag 側
#KANKO_DB_DIR = r"C:/Users/harunobu/Desktop/okazaki_rag/chroma_db"  # ★観光側（必要なら書換）

BASE_WASTE = Path(__file__).resolve().parents[1]
PORTAL_ROOT = BASE_WASTE.parents[0]

# ===== コレクション名 =====
COLLECTIONS = {
    "ごみ": {"db": "waste", "name": "okazaki_waste"},
    "暮らし": {"db": "waste", "name": "okazaki_life"},
    # ★ここは list_collections.py の結果で確定させる（例: okazaki_events）
    "観光": {"db": "kanko", "name": "okazaki_events"},
}

FAQ_COLLECTIONS = {
    # cat: list of (db_kind, collection_name, stage_name)
    "ごみ": [
        ("waste", "okazaki_waste_faq", "faq"),
    ],
    "観光": [
        # ★優先順位：固定FAQ → イベントFAQ → main
        ("kanko", "okazaki_events_fixed_faq", "faq_fixed"),
        ("kanko", "okazaki_events_faq", "faq_events"),
    ],
    # "暮らし": [("waste", "okazaki_life_faq", "faq")],  # 将来追加するなら
}

# FAQ 段ごとのデフォルト（必要なら後で normal60 で調整）
FAQ_STAGE_DEFAULTS = {
    # 固定情報は短文・確定情報なので「やや厳しめ」でOK（誤答ゼロ優先）
    "faq_fixed": {"top_k": 5, "dist_th": 1.15},
    # イベントFAQは表現ゆれが出やすいので「やや緩め」
    "faq_events": {"top_k": 5, "dist_th": 1.30},
    # 既存（ごみ等）
    "faq": {"top_k": 5, "dist_th": 1.30},
}

def get_faq_collections(waste_client, kanko_client, cat: str):
    """
    Returns: list[dict]
      [{
        "col": chroma_collection | None,
        "name": collection_name,
        "stage": stage_name,  # e.g. faq_fixed / faq_events / faq
      }, ...]
    """
    specs = FAQ_COLLECTIONS.get(cat) or []
    out = []
    for (db_kind, name, stage) in specs:
        cli = waste_client if db_kind == "waste" else kanko_client
        try:
            out.append({"col": cli.get_collection(name), "name": name, "stage": stage})
        except Exception:
            out.append({"col": None, "name": name, "stage": stage})
    return out

DIST_THRESHOLD_BY_CAT = {
    "ごみ": 1.20,
    "暮らし": 1.30,
    "観光": 1.80,
}
FAQ_DIST_THRESHOLD_BY_CAT = {
    "ごみ": 1.30,   # FAQは短文なので main より少し緩くしてOK
}
FAQ_TOP_K_BY_CAT = {
    "ごみ": 5,
}

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"
TOP_K = 6
TOP_K_BY_CAT = {"ごみ": 12, "暮らし": 12, "観光": 6}
DIST_THRESHOLD = 1.10  # best distance がこれより大きい場合は回答停止（要調整）


SYSTEM_PROMPT = """あなたは岡崎市の公式資料（PDF/公開データ等）に基づいて回答するアシスタントです。
次を必ず守ってください。

- 提示された「参考資料（抜粋）」に書かれていない内容は推測で断定しない。
- 回答の最後に必ず「根拠（URL / ページ）」を箇条書きで列挙する。
- 根拠が不十分な場合は「資料内で確認できません」と明示し、公式確認を促す。
- 回答は結論→手順/注意点（箇条書き）で簡潔に。
"""

ROUTER_PROMPT = """あなたはユーザーの質問を、次のカテゴリのどれか1つに分類してください。
カテゴリは ["ごみ", "暮らし", "観光"] のみです。

分類ルール:
- ごみ: 分別、出し方、指定袋、収集、リサイクル、粗大ごみ等
- 暮らし: 転入転出、届出、税金、保険年金、子育て、福祉、施設案内、相談窓口、防災（生活系）等
- 観光: イベント、観光スポット、アクセス、モデルコース、食事、宿泊、観光客向け案内等

出力はカテゴリ名のみ（例: 観光）。説明は不要。
"""

def build_context(res) -> tuple[str, list[str]]:
    """Retrieve 文書抜粋と根拠(URL/page)のリストを生成する。

    重要:
    - FAQ（Q/A）由来の重複が出やすいので、まず「Q:」の先頭行で重複排除する
    - それでも取れない場合は、空白正規化した全文で重複排除する
    """
    docs = res.get("documents", [[]])[0] or []
    metas = res.get("metadatas", [[]])[0] or []

    blocks: list[str] = []
    cites: list[str] = []

    seen_q: set[str] = set()
    seen_text: set[str] = set()

    for i, d in enumerate(docs):
        if not d:
            continue
        b = str(d).strip()

        # --- 1) FAQ重複排除: 先頭の Q: 行（最初の1行）で潰す ---
        qkey = None
        for line in b.splitlines():
            line = line.strip()
            if line.startswith("Q:"):
                qkey = re.sub(r"\s+", " ", line[2:].strip())
                qkey = re.sub(r"[　 ]+", " ", qkey)
                break
        if qkey:
            if qkey in seen_q:
                continue
            seen_q.add(qkey)

        # --- 2) それ以外: 全文を空白正規化して重複排除 ---
        tkey = re.sub(r"\s+", " ", b).strip()
        if tkey in seen_text:
            continue
        seen_text.add(tkey)

        blocks.append(b)

        # --- cites ---
        m0 = metas[i] or {}
        url = (
            m0.get("pdf_url")
            or m0.get("file_url")
            or m0.get("apply_url")
            or m0.get("source_url")
            or m0.get("source")
            or ""
        )
        page = m0.get("page")
        if url:
            if page is not None and str(page).strip() != "":
                cites.append(f"{url} (p.{page})")
            else:
                cites.append(str(url))

    # cites も重複排除（順序維持）
    uniq_cites: list[str] = []
    seen_c: set[str] = set()
    for c in cites:
        if c in seen_c:
            continue
        seen_c.add(c)
        uniq_cites.append(c)

    ctx_parts = []
    for j, b in enumerate(blocks):
        ctx_parts.append(f"[資料{j+1}]\n{b}")
    ctx = "\n\n".join(ctx_parts).strip()

    return ctx, uniq_cites

def get_best_distance(res: Dict[str, Any]) -> float:
    try:
        dists = res.get("distances", [[]])[0]
        if not dists:
            return 999.0
        return float(min(dists))
    except Exception:
        return 999.0


def route_category(oa: OpenAI, user_query: str) -> str:
    r = oa.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": user_query},
        ],
        temperature=0.0,
    )
    cat = (r.choices[0].message.content or "").strip()
    return cat if cat in ["ごみ", "暮らし", "観光"] else "暮らし"
def filter_hits_by_domain(hits, deny_domains):
    out = []
    for h in hits:
        url = (h.get("url") or h.get("pdf_url") or "")
        if any(d in url for d in deny_domains):
            continue
        out.append(h)
    return out

def rule_based_category(q: str) -> str | None:
    """軽量ルータ（ルール優先）。

    目的: ありがちな誤分類（例: 「今日」だけでごみ/暮らしに寄る、桜/開花が別カテゴリに流れる）を減らす。
    """
    waste_kw = ["ごみ", "ゴミ", "分別", "何ごみ", "収集", "指定ごみ袋", "指定ゴミ袋", "粗大", "リサイクル", "電池", "モバイルバッテリー"]
    life_kw  = ["転入", "転出", "住民票", "マイナンバー", "国保", "年金", "税", "保育", "福祉", "手続き"]
    kanko_kw = [
        "観光", "イベント", "祭り", "岡崎城", "モデルコース", "ランチ", "宿泊", "アクセス",

        # 桜・花見（3月〜4月の誤分類防止）
        "桜", "花見", "お花見", "開花", "満開", "見頃", "夜桜", "ライトアップ",

        # 当日変動で観光に寄せたい語
        "交通規制", "通行規制", "通行止め", "迂回",
        "待ち時間", "入場待ち", "混雑",
        "臨時駐車場", "臨時駐輪場", "駐輪場",
        "最新情報",
    ]

    if any(k in q for k in waste_kw):
        return "ごみ"
    if any(k in q for k in life_kw):
        return "暮らし"
    if any(k in q for k in kanko_kw):
        return "観光"
    return None

# =========================
# 観光：固定情報 intent 判定
# =========================
TOURISM_FIXED_INFO_KW = [
    "営業時間", "開館", "開館時間", "休館", "定休日",
    "料金", "観覧料", "入場", "入館", "入館料", "入園料", "チケット",
    "住所", "所在地", "場所", "どこ",
    "公式サイト", "ホームページ", "URL", "リンク",
    "電話", "連絡先", "問い合わせ",
    "アクセス", "行き方", "最寄り駅",
    "駐車場", "バス駐車場",
    "所要時間", "見学目安", "回る時間", "滞在時間",
    "バリアフリー", "車いす", "車椅子", "多目的トイレ",
]

def is_tourism_fixed_info_query(q: str) -> bool:
    q = (q or "")
    return any(k in q for k in TOURISM_FIXED_INFO_KW)

TOURISM_VOLATILE_EXTRA_KW = [
    # 時間・速報系
    "今日", "本日", "現在", "いま", "今", "今週", "今週末",
    "最新", "最新情報", "直近", "きょう",

    # 混雑・待ち系
    "混雑", "待ち時間", "入場待ち",

    # 花・見頃系
    "開花", "満開", "見頃", "咲き具合",

    # 開催状況系
    "開催中", "実施", "中止", "延期",

    # 予約・空席系
    "予約枠", "予約状況", "空き", "空席", "空き状況", "空いていますか",

    # 規制・交通系
    "通行規制", "通行止め", "迂回", "交通規制", "規制情報",

    # 工事・閉鎖系
    "工事", "工事中", "閉鎖", "一部閉鎖", "利用停止",

    # 営業可否系
    "閉館", "休園", "臨時休園", "臨時休館",

    # 特別運用
    "ライトアップ",
]

def is_tourism_volatile_query_extra(q: str) -> bool:
    q = (q or "")

    # まず単純一致
    if any(k in q for k in TOURISM_VOLATILE_EXTRA_KW):
        return True

    # 時間語 × 状態語 の組み合わせでも volatile 扱い
    time_words = [
        "今日", "本日", "現在", "いま", "今", "今週", "今週末", "最新", "直近"
    ]
    state_words = [
        "空いて", "空き", "空席", "予約", "予約状況",
        "混雑", "待ち時間",
        "開催", "実施", "中止", "延期",
        "休園", "休館", "閉館", "閉鎖",
        "工事", "工事中",
        "通行規制", "通行止め", "迂回", "規制"
    ]

    if any(t in q for t in time_words) and any(s in q for s in state_words):
        return True

    return False

TOURISM_GUIDE_FALLBACK_KW = [
    "見どころ", "特徴", "展示",
    "所要時間", "見学目安", "回る時間",
    "おすすめ", "優先ポイント", "回り方",
    "短時間で", "半日で", "モデルコース",
]

def is_tourism_guide_query(q: str) -> bool:
    q = (q or "")
    return any(k in q for k in TOURISM_GUIDE_FALLBACK_KW)

def get_clients():
    #waste_client = chromadb.PersistentClient(
    #    path=WASTE_LIFE_DB_DIR, settings=Settings(allow_reset=False)
    #)
    waste_client = chromadb.PersistentClient(path=str(WASTE_LIFE_DB_DIR))
    #st.write("CWD:", os.getcwd())
    #st.write("WASTE_LIFE_DB_DIR(abs):", str(Path(WASTE_LIFE_DB_DIR).resolve()))
    #st.write("KANKO_DB_DIR(abs):", str(Path(KANKO_DB_DIR).resolve()))
    kanko_client = chromadb.PersistentClient(path=str(KANKO_DB_DIR))
    #kanko_client = chromadb.PersistentClient(
    #    path=KANKO_DB_DIR, settings=Settings(allow_reset=False)
    #)
    return waste_client, kanko_client

def pick_collection(waste_client, kanko_client, category: str):
    cfg = COLLECTIONS[category]
    client = waste_client if cfg["db"] == "waste" else kanko_client
    return client.get_collection(cfg["name"]), cfg["name"], cfg["db"]

def answer_portal(
    oa: OpenAI,
    waste_client,
    kanko_client,
    user_query: str,
    *,
    year: int | None = None,
    forced_cat: str | None = None,
) -> tuple[str, list[str], str, str, str, str, float]:
    
    # --- 前回値の持ち越し防止（ログ汚染を防ぐ） ---
    st.session_state["tourism_dec_reason"] = None
    st.session_state["tourism_dec_hit"] = None
    st.session_state["tourism_is_fixed_info"] = None

    """
    Returns:
      ans, cites, ctx, status, chosen_cat, chosen_collection, best_dist
    """

     # 1) カテゴリ決定（優先：forced → 観光volatile即時固定 → ルール → LLM）
    if forced_cat in ["ごみ", "暮らし", "観光"]:
        cat = forced_cat
    elif is_tourism_volatile_query_extra(user_query):
        # 観光の当日変動系は、ルータ誤分類前に観光へ固定する
        cat = "観光"
    else:
        cat = rule_based_category(user_query) or route_category(oa, user_query)

    # 2) 第1段階：そのカテゴリで検索
    col, col_name, _db_kind = pick_collection(waste_client, kanko_client, cat)

    faq_specs = get_faq_collections(waste_client, kanko_client, cat)

    where = {"year": int(year)} if (cat == "観光" and year is not None) else None

    ans1, cites1, ctx1, status1, dist1 = answer_with_collection(
        oa, col, user_query, category=cat, where=where,
        faq_specs=faq_specs
    )

    # 観光の「ルールabstain」は横断検索に進ませない（事故防止）
    if cat == "観光" and status1 == "abstain" and st.session_state.get("abstain_reason") == "tourism_rule":
        return ans1, cites1, ctx1, "abstain", cat, col_name, dist1

    if status1 == "ok" and dist1 <= DIST_THRESHOLD:
        return ans1, cites1, ctx1, "ok", cat, col_name, dist1

    # 3) 第2段階：横断検索（安全設計）
    # ごみ/暮らしのときは観光を横断しない（イベント混入の根治）
    if cat in ("ごみ", "暮らし"):
        cats2 = ["ごみ", "暮らし"]
    else:
        cats2 = ["ごみ", "暮らし", "観光"]

    best = None

    for c in cats2:
        c_col, c_name, _ = pick_collection(waste_client, kanko_client, c)
        c_where = {"year": int(year)} if (c == "観光" and year is not None) else None

        c_faq_specs = get_faq_collections(waste_client, kanko_client, c)
        c_ans, c_cites, c_ctx, c_status, c_dist = answer_with_collection(
            oa, c_col, user_query, category=c, where=c_where,
            faq_specs=c_faq_specs
        )

        if c_status == "ok" and c_dist <= DIST_THRESHOLD:
            if best is None or c_dist < best[0]:
                best = (c_dist, c, c_name, c_ans, c_cites, c_ctx)

    if best is not None:
        best_dist, chosen_cat, chosen_col, ans, cites, ctx = best
        return ans, cites, ctx, "ok", chosen_cat, chosen_col, best_dist

    # 4) 全滅：第1段階の結果（abstain）をそのまま返す（未定義を絶対作らない）
    return ans1, cites1, ctx1, status1, cat, col_name, dist1

def rerank_hits(res: dict) -> dict:
    """
    Chroma query結果 res を、PDF根拠を優先するように再ランキングして返す。
    前提: distances は「小さいほど良い」
    """
    docs0  = res.get("documents", [[]])[0]
    metas0 = res.get("metadatas", [[]])[0]
    dists0 = res.get("distances", [[]])[0]

    if not docs0 or not metas0 or not dists0:
        return res

    scored = []
    for i, (doc, meta, dist) in enumerate(zip(docs0, metas0, dists0)):
        m = meta or {}
        url = (m.get("pdf_url") or m.get("file_url") or m.get("source") or "")
        title = (m.get("title") or m.get("file_name") or "")

        # --- 減点・加点（distに足し引き）---
        adj = float(dist)

        # 1) PDF優先：URLが .pdf / doc_type が pdf系ならボーナス（= distを小さく）
        doc_type = (m.get("doc_type") or "").lower()
        if ".pdf" in (url.lower()) or "pdf" in doc_type:
            adj -= 0.25  # 強め。効きすぎるなら -0.15 などに

        # 2) イベントCSV（bodikのイベントリソース）を強くペナルティ
        if "data.bodik.jp" in url and "download" in url:
            adj += 0.60

        # 3) タイトルがイベントっぽい語なら少しペナルティ（保険）
        if any(k in title for k in ["お花見", "マルシェ", "フェス", "ラリー", "グランプリ"]):
            adj += 0.30

        scored.append((adj, i))

    scored.sort(key=lambda x: x[0])

    # rerank_hits の最後（置き換え）
    order = [i for _, i in scored]
    adj_dists_by_i = {i: adj for (adj, i) in scored}

    res["documents"] = [[docs0[i] for i in order]]
    res["metadatas"] = [[metas0[i] for i in order]]
    res["distances"] = [[float(adj_dists_by_i[i]) for i in order]]  # ←ここが重要
    return res

def answer_with_collection(
    oa: OpenAI,
    col,
    user_query: str,
    category: str,
    *,
    where: Dict[str, Any] | None = None,
    faq_specs: List[Dict[str, Any]] | None = None,
) -> Tuple[str, List[str], str, str, float]:
    """
    Returns:
      ans: 回答 or フォールバック文
      cites: 自動抽出根拠
      context_text: 抜粋
      status: "ok" or "abstain"
      best_dist: 採用した検索の最良distance（小さいほど良い）
    """

    # --- ログ用の最小メトリクス（UI側でappend_logする） ---
    st.session_state["answered_by"] = "main"
    st.session_state["main_best_dist"] = None
    st.session_state["faq_best_dist"] = None
    st.session_state["faq_stage"] = None
    st.session_state["faq_stage_best"] = {}
    st.session_state["faq_query_error"] = None
    st.session_state["collection_used"] = "main"
    st.session_state["abstain_reason"] = None
    st.session_state["abstain_hit"] = None

    q_emb = oa.embeddings.create(model=EMBED_MODEL, input=[user_query]).data[0].embedding

    NG_PHRASES = (
        "見当たりません",
        "記載がありません",
        "含まれていません",
        "明確な記述がありません",
        "資料内で確認できません",
    )
    # =========================================
    # 0) 観光：当日変動系はFAQ生成前に強制abstain（最優先）
    # =========================================

    # distゲート（幻覚防止）
    threshold = DIST_THRESHOLD_BY_CAT.get(category, 1.10)
    # 観光の「固定情報」は拾い上げたいので、固定情報intentだけ閾値を緩める
    tourism_fixed = (category == "観光" and is_tourism_fixed_info_query(user_query))
    tourism_fixed_threshold = 1.20  # ←まずここから。後で normal60 で調整

    # =========================================
    # 0) 観光：当日変動系はFAQ生成前に強制abstain（最優先）
    # =========================================
    if category == "観光":
        # app側の追加キーワードで当日変動を先に止める
        if is_tourism_volatile_query_extra(user_query):
            st.session_state["answered_by"] = "abstain"
            st.session_state["collection_used"] = "tourism_abstain_rule"
            st.session_state["abstain_reason"] = "tourism_rule"
            st.session_state["abstain_hit"] = "extra_keyword"
            st.session_state["tourism_dec_reason"] = "keyword"
            st.session_state["tourism_dec_hit"] = "extra_keyword"
            st.session_state["tourism_is_fixed_info"] = False
            return TOURISM_ABSTAIN_MESSAGE, [], "", "abstain", 0.0

        # best_dist 未取得の段階では 0.0 を渡し、
        # 「今日/今/混雑/開花/見頃」など keyword 起因だけを先に止める
        dec0 = decide_tourism_abstain(user_query, 0.0, threshold)
        if os.getenv("DEBUG_ABSTAIN") == "1":
            st.write({
                "tourism_abstain_precheck": {
                    "abstain": dec0.abstain,
                    "reason": dec0.reason,
                    "hit": dec0.hit,
                    "best_dist": 0.0,
                    "threshold": threshold,
                }
            })
        if dec0.abstain and dec0.reason == "keyword":
            st.session_state["answered_by"] = "abstain"
            st.session_state["collection_used"] = "tourism_abstain_rule"
            st.session_state["abstain_reason"] = "tourism_rule"
            st.session_state["abstain_hit"] = dec0.hit
            st.session_state["tourism_dec_reason"] = dec0.reason
            st.session_state["tourism_dec_hit"] = dec0.hit
            st.session_state["tourism_is_fixed_info"] = False
            return TOURISM_ABSTAIN_MESSAGE, [], "", "abstain", 0.0
 
    # =========================================
    # 0) FAQ優先
    # =========================================
    faq_specs = faq_specs or []
    for spec in faq_specs:
        faq_col = spec.get("col")
        faq_name = spec.get("name")
        faq_stage = spec.get("stage") or "faq"
        if faq_col is None:
            continue

        cfg = FAQ_STAGE_DEFAULTS.get(faq_stage, FAQ_STAGE_DEFAULTS["faq"])
        faq_n = int(cfg["top_k"])
        faq_th = float(cfg["dist_th"])
        try:
            faq_res = faq_col.query(
                query_embeddings=[q_emb],
                n_results=faq_n,
                include=["documents", "metadatas", "distances"],
            )
            faq_res = rerank_hits(faq_res)
        except Exception as e:
            # FAQコレクションの次元不一致などでは main へフォールバック
            st.session_state["faq_stage_best"][faq_stage] = "query_error"
            st.session_state["faq_query_error"] = f"{faq_name}: {type(e).__name__}: {e}"
            continue

        try:
            faq_best = float(faq_res.get("distances", [[]])[0][0])
        except Exception:
            faq_best = 999.0

        # 段別の best_dist を保持（デバッグ/レポート用）
        st.session_state["faq_stage_best"][faq_stage] = faq_best

        if faq_best <= faq_th:
            faq_ctx, faq_cites = build_context(faq_res)

            user_prompt = f"""質問: {user_query}

参考資料（抜粋）:
{faq_ctx}

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
            faq_ans = (chat.choices[0].message.content or "").strip()

            if not any(p in faq_ans for p in NG_PHRASES):
                # ★どの段で当たったかをログに残す
                st.session_state["answered_by"] = faq_stage
                st.session_state["faq_stage"] = faq_stage
                st.session_state["faq_best_dist"] = faq_best
                st.session_state["collection_used"] = (faq_name or faq_stage)
                return faq_ans, faq_cites, faq_ctx, "ok", faq_best
        # NG もしくは閾値超えなら、次段（固定→イベント）へ

    # =========================
    # 1) main（FAQがダメなら本体）
    # =========================
    n = TOP_K_BY_CAT.get(category, TOP_K)
    res = col.query(
        query_embeddings=[q_emb],
        n_results=n,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    res = rerank_hits(res)

    # debug hits（現行維持）
    try:
        d0 = res.get("distances", [[]])[0]
        m0 = res.get("metadatas", [[]])[0]
        rows = []
        for i in range(min(5, len(d0))):
            title = (m0[i] or {}).get("title") or (m0[i] or {}).get("file_name") or ""
            url = (
                (m0[i] or {}).get("pdf_url")
                or (m0[i] or {}).get("file_url")
                or (m0[i] or {}).get("source")
                or ""
            )
            rows.append({"rank": i+1, "dist": float(d0[i]), "title": title, "url": url})
        st.write(f"debug hits: category={category} where={where}", rows)
    except Exception:
        pass

    context_text, cites = build_context(res)

    try:
        best_dist = float(res.get("distances", [[]])[0][0])
    except Exception:
        best_dist = 999.0

    st.session_state["main_best_dist"] = best_dist
    st.session_state["collection_used"] = "main"

    # mainのabstainメッセージ
    main_abstain_msg = (
        "資料内で根拠を確認できなかったため、推測で回答できません。"
        "次のように具体化して再度質問してください："
        "・制度名 / 手続き名 / 対象者（年齢・状況）"
        "・年度（例：2025年、2026年）"
        "・場所（施設名・窓口名）"
    )

    # distゲート（幻覚防止）
    #threshold = DIST_THRESHOLD_BY_CAT.get(category, 1.10)

    # =========================
    # 観光専用 abstain（文意優先 + dist）
    # =========================
    if category == "観光":
        # 固定情報intentは dist で落とさない（落とすのは「当日変動 keyword」のみ）
        th_for_decision = (tourism_fixed_threshold if tourism_fixed else threshold)
        dec = decide_tourism_abstain(user_query, best_dist, th_for_decision)
                # --- ログ可視化用（観光intent情報） ---
        st.session_state["tourism_dec_reason"] = dec.reason
        st.session_state["tourism_dec_hit"] = dec.hit
        st.session_state["tourism_is_fixed_info"] = (isinstance(dec.hit, str) and dec.hit.startswith("fixed:"))

        if os.getenv("DEBUG_ABSTAIN") == "1":
            st.write({"tourism_abstain_decision": {"abstain": dec.abstain, "reason": dec.reason, "hit": dec.hit, "best_dist": best_dist, "threshold": threshold}})
        if dec.abstain:
            st.session_state["answered_by"] = "abstain"
            st.session_state["collection_used"] = "tourism_abstain_rule"
            st.session_state["abstain_reason"] = "tourism_rule"
            st.session_state["abstain_hit"] = dec.hit

            # keyword だけは即abstain
            if dec.reason == "keyword":
                msg = (
                    "その内容は当日の状況（天候・運営・混雑など）によって変わるため、"
                    "この案内では正確にお答えできません。\n\n"
                    "最新情報は公式サイト（お知らせ/イベントページ）をご確認ください。"
                )
                return msg, [], "", "abstain", best_dist

            # dist / softdist は即abstainしない
            # → main_pass 判定と回答生成へ進ませる
            st.session_state["answered_by"] = "main"
            st.session_state["collection_used"] = "main"
            st.session_state["abstain_reason"] = None
            st.session_state["abstain_hit"] = None          


    if category == "観光" and tourism_fixed:
        main_pass = (best_dist <= tourism_fixed_threshold)
    else:
        main_pass = (best_dist <= threshold)

    # 「購入場所」系の安全ガード（mainのときだけ）
    BUY_WORDS = ("買える", "販売", "売って", "購入", "どこで買", "取扱", "取り扱い", "入手")
    if main_pass and category == "ごみ" and any(w in user_query for w in BUY_WORDS):
        if (not cites) or (not any("http" in c for c in cites)):
            msg = (
                "購入場所（どこで買える/販売店）については、資料内で根拠を確認できなかったため回答できません。"
                "岡崎市の指定ごみ袋の取扱い（販売店・回収協力店など）は、公式資料の該当ページでご確認ください。"
            )
            return msg, cites, context_text, "abstain", best_dist

    # mainで回答生成（main_pass のときだけ）
    if main_pass:
        user_prompt = f"""質問: {user_query}

参考資料（抜粋）:
{context_text}

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

        if not any(p in ans for p in NG_PHRASES):
            st.session_state["answered_by"] = "main"
            return ans, cites, context_text, "ok", best_dist

        # =========================
        # 観光：固定情報intentの最終フォールバック（リンク提示）
        # =========================
        if category == "観光" and tourism_fixed:
            if cites:
                msg = (
                    "該当情報をこのデータベース内で十分に抜粋できませんでした。"
                    "ただし、関連する公式ページのURLは取得できたため、まずこちらをご確認ください。\n\n"
                    "※必要なら、URLを開いた上で「営業時間」「料金」「住所」など知りたい項目を指定して再質問してください。"
                )
                # これは abstain ではなく「回答（ok）」扱いにする（abstain削減のため）
                return msg, cites, context_text, "ok", best_dist        

    # =========================================================
    # ★ここが欠けていた：main_pass=False でも必ず返す（None防止）
    # =========================================================
    # 観光の固定情報は、main_pass に落ちても「リンク提示」で ok にできるなら ok にする
    if category == "観光" and tourism_fixed and cites:
        msg = (
            "該当情報をこのデータベース内で十分に抜粋できませんでした。"
            "ただし、関連する公式ページのURLは取得できたため、まずこちらをご確認ください。\n\n"
            "※必要なら、URLを開いた上で「営業時間」「料金」「住所」など知りたい項目を指定して再質問してください。"
        )
        return msg, cites, context_text, "ok", best_dist
    
        # 観光ガイド系は、当日変動でなく cites があるならリンク案内で ok にする
    if category == "観光" and is_tourism_guide_query(user_query) and cites:
        msg = (
            "この質問に対して、資料内の記述だけで断定的な案内文を組み立てるには根拠が不足していました。"
            "ただし、関連する公式ページのURLは取得できたため、まずこちらをご確認ください。\n\n"
            "※必要なら、URLを開いた上で「展示内容」「所要時間」「見どころ」など項目を指定して再質問してください。"
        )
        st.session_state["answered_by"] = "guide_fallback"
        st.session_state["collection_used"] = "guide_fallback"
        return msg, cites, context_text, "ok", best_dist

    # それ以外は abstain（根拠不足）
    return main_abstain_msg, cites, context_text, "abstain", best_dist   

def extract_top_sources(cites: list[str] | None, limit: int = 3) -> list[dict]:
    """
    cites 文字列から、最低限の根拠情報を構造化して残す。
    うまく取れない場合でも url だけ残す。
    """
    out = []
    if not cites:
        return out

    for c in cites:
        if len(out) >= limit:
            break
        s = (c or "").strip()

        # 雑にURL抽出（最小）
        url = ""
        for token in s.split():
            if token.startswith("http://") or token.startswith("https://"):
                url = token
                break

        # ページらしき数字があれば拾う（例: "p.12" "ページ 12"）
        page = None
        # 超簡易：数字の連続を拾う（過剰マッチしうるが最小実装）
        digits = "".join(ch if ch.isdigit() else " " for ch in s).split()
        if digits:
            try:
                page = int(digits[-1])
            except Exception:
                page = None

        out.append({"raw": s[:200], "url": url, "page": page})

    return out

def append_log(path: str, record: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def get_suite_from_url(default: str = "manual") -> str:
    """
    URLクエリ ?suite=beta_fixed のような値を取得する。
    Streamlitのバージョン差異に備えて、
    st.query_params と st.experimental_get_query_params の両方に対応する。
    """
    try:
        # 新しめのStreamlit
        qp = st.query_params
        val = qp.get("suite", default)

        # st.query_params は list ではなく単一値になることが多いが、
        # 念のため list/tuple にも対応
        if isinstance(val, (list, tuple)):
            val = val[0] if val else default

        val = str(val).strip()
        return val if val else default

    except Exception:
        pass

    try:
        # 旧Streamlit互換
        qp = st.experimental_get_query_params()
        val = qp.get("suite", [default])
        if isinstance(val, list):
            val = val[0] if val else default
        val = str(val).strip()
        return val if val else default
    except Exception:
        return default

def get_run_id_from_url(default: str = "") -> str:
    """
    URLクエリ ?run_id=beta_volatile_20260314_01 のような値を取得する。
    """
    try:
        qp = st.query_params
        val = qp.get("run_id", default)
        if isinstance(val, (list, tuple)):
            val = val[0] if val else default
        val = str(val).strip()
        return val if val else default
    except Exception:
        pass

    try:
        qp = st.experimental_get_query_params()
        val = qp.get("run_id", [default])
        if isinstance(val, list):
            val = val[0] if val else default
        val = str(val).strip()
        return val if val else default
    except Exception:
        return default
def inject_admin_ui_css() -> None:
    st.markdown("""
    <style>
    :root {
        --okz-blue: #0b57a4;
        --okz-blue-2: #1f6fc2;
        --okz-sky: #f4f8fc;
        --okz-border: #d7e3f1;
        --okz-text: #1f2937;
        --okz-sub: #5b6573;
        --okz-warn: #fff7e6;
        --okz-warn-border: #f3d39a;
        --okz-card: #ffffff;
        --okz-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        --okz-radius: 16px;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    .okz-hero {
        background: linear-gradient(135deg, #0b57a4 0%, #1f6fc2 100%);
        color: white;
        padding: 24px 24px 20px 24px;
        border-radius: 20px;
        box-shadow: var(--okz-shadow);
        margin-bottom: 18px;
    }

    .okz-hero-badge {
        display: inline-block;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: .04em;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.16);
        margin-bottom: 10px;
    }

    .okz-hero-title {
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.3;
        margin: 0 0 8px 0;
    }

    .okz-hero-desc {
        font-size: 0.98rem;
        line-height: 1.8;
        opacity: 0.98;
        margin: 0;
    }

    .okz-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin: 14px 0 18px 0;
    }

    .okz-card {
        background: var(--okz-card);
        border: 1px solid var(--okz-border);
        border-radius: var(--okz-radius);
        box-shadow: var(--okz-shadow);
        padding: 16px 16px 14px 16px;
    }

    .okz-card-title {
        color: var(--okz-blue);
        font-size: 1rem;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .okz-card-body {
        color: var(--okz-text);
        font-size: 0.93rem;
        line-height: 1.75;
    }

    .okz-chip-row {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 8px;
    }

    .okz-chip {
        display: inline-block;
        border: 1px solid var(--okz-border);
        background: #fff;
        color: var(--okz-blue);
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 12px;
        font-weight: 700;
    }

    .okz-note {
        background: var(--okz-warn);
        border: 1px solid var(--okz-warn-border);
        border-radius: 14px;
        padding: 14px 16px;
        margin: 12px 0 18px 0;
        color: #5f4b1c;
        font-size: 0.93rem;
        line-height: 1.7;
    }

    .okz-section-title {
        font-size: 1.1rem;
        font-weight: 800;
        color: var(--okz-blue);
        margin: 18px 0 10px 0;
    }

    .okz-footer {
        margin-top: 28px;
        padding: 18px 18px 8px 18px;
        border-top: 1px solid var(--okz-border);
        color: var(--okz-sub);
        font-size: 0.9rem;
        line-height: 1.8;
    }

    .okz-side-box {
        background: #ffffff;
        border: 1px solid var(--okz-border);
        border-radius: 14px;
        padding: 12px 12px 10px 12px;
        margin-bottom: 12px;
    }

    .okz-side-title {
        font-size: 0.92rem;
        font-weight: 800;
        color: var(--okz-blue);
        margin-bottom: 6px;
    }

    .okz-side-text {
        font-size: 0.84rem;
        color: var(--okz-text);
        line-height: 1.7;
    }

    .stTextInput > div > div > input,
    .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 12px !important;
    }

    .stButton > button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        min-height: 44px !important;
    }
    .okz-answer-card {
        background: #ffffff;
        border: 1px solid var(--okz-border);
        border-left: 6px solid var(--okz-blue);
        border-radius: 18px;
        box-shadow: var(--okz-shadow);
        padding: 18px 18px 14px 18px;
        margin: 14px 0 16px 0;
    }

    .okz-answer-label {
        display: inline-block;
        background: #eaf3ff;
        color: var(--okz-blue);
        border-radius: 999px;
        padding: 5px 10px;
        font-size: 12px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .okz-answer-text {
        color: var(--okz-text);
        font-size: 1rem;
        line-height: 1.9;
        white-space: pre-wrap;
    }

    .okz-proof-card {
        background: #ffffff;
        border: 1px solid var(--okz-border);
        border-radius: 16px;
        box-shadow: var(--okz-shadow);
        padding: 14px 16px;
        margin: 10px 0 12px 0;
    }

    .okz-proof-title {
        font-size: 0.95rem;
        font-weight: 800;
        color: var(--okz-blue);
        margin-bottom: 8px;
    }

    .okz-meta-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        margin: 10px 0 14px 0;
    }

    .okz-meta-box {
        background: #fff;
        border: 1px solid var(--okz-border);
        border-radius: 12px;
        padding: 10px 12px;
    }

    .okz-meta-label {
        font-size: 0.75rem;
        color: var(--okz-sub);
        margin-bottom: 4px;
    }

    .okz-meta-value {
        font-size: 0.92rem;
        font-weight: 700;
        color: var(--okz-text);
        word-break: break-word;
    }
        .okz-input-wrap {
        background: #ffffff;
        border: 1px solid var(--okz-border);
        border-radius: 18px;
        box-shadow: var(--okz-shadow);
        padding: 16px;
        margin: 12px 0 18px 0;
    }

    .okz-input-title {
        font-size: 1rem;
        font-weight: 800;
        color: var(--okz-blue);
        margin-bottom: 8px;
    }

    .okz-input-help {
        font-size: 0.9rem;
        color: var(--okz-sub);
        margin-bottom: 12px;
        line-height: 1.7;
    }

    .okz-preset-title {
        font-size: 0.95rem;
        font-weight: 800;
        color: var(--okz-blue);
        margin: 12px 0 8px 0;
    }

    div[data-testid="stTextInput"] input {
        min-height: 48px !important;
        font-size: 16px !important;
    }

    div[data-testid="stForm"] {
        margin-top: 8px;
    }
    .okz-send-btn button {
    background: #2563eb !important;
    color: white !important;
    border-radius: 14px !important;
    height: 48px !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    border: none !important;
    }

    .okz-send-btn button:hover {
        background: #1e4fd6 !important;
    }

    .okz-send-btn button:active {
        background: #1a44b8 !important;
    }
    .okz-loading-note {
        background: #eef6ff;
        border: 1px solid #cfe1fb;
        border-radius: 14px;
        padding: 12px 14px;
        margin: 10px 0 14px 0;
        color: #1f4f8f;
        font-size: 0.92rem;
        line-height: 1.7;
    }            
    @media (max-width: 900px) {
        .okz-meta-grid {
            grid-template-columns: 1fr;
        }
    }
    @media (max-width: 900px) {
        .okz-grid {
            grid-template-columns: 1fr;
        }
        .okz-hero {
            padding: 18px 16px 16px 16px;
        }
        .okz-hero-title {
            font-size: 1.5rem;
        }
        .block-container {
            padding-top: 0.8rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def render_admin_header() -> None:
    st.markdown("""
    <div class="okz-hero">
        <div class="okz-hero-badge">β版 / 岡崎市向け実証中</div>
        <div class="okz-hero-title">岡崎市AI総合案内</div>
        <p class="okz-hero-desc">
            岡崎市に関する <b>ごみ・暮らし・観光</b> の情報を、公式資料を根拠として案内するAIです。<br>
            質問内容に応じてカテゴリを自動判定し、該当データを検索します。
        </p>
    </div>
    """, unsafe_allow_html=True)
def render_quick_guide() -> None:
    st.markdown('<div class="okz-section-title">このサービスでできること</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="okz-grid">
        <div class="okz-card">
            <div class="okz-card-title">ごみ</div>
            <div class="okz-card-body">
                分別方法、出し方、指定ごみ袋、粗大ごみ、資源回収などを案内します。
                <div class="okz-chip-row">
                    <span class="okz-chip">分別</span>
                    <span class="okz-chip">収集</span>
                    <span class="okz-chip">指定袋</span>
                </div>
            </div>
        </div>
        <div class="okz-card">
            <div class="okz-card-title">暮らし</div>
            <div class="okz-card-body">
                手続き、届出、保険・年金、子育て、福祉、窓口情報などを案内します。
                <div class="okz-chip-row">
                    <span class="okz-chip">転入転出</span>
                    <span class="okz-chip">住民票</span>
                    <span class="okz-chip">窓口</span>
                </div>
            </div>
        </div>
        <div class="okz-card">
            <div class="okz-card-title">観光</div>
            <div class="okz-card-body">
                観光施設、アクセス、モデルコース、固定情報を中心に案内します。
                当日変動情報は回答を控える場合があります。
                <div class="okz-chip-row">
                    <span class="okz-chip">施設情報</span>
                    <span class="okz-chip">アクセス</span>
                    <span class="okz-chip">モデルコース</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="okz-note">
        <b>ご利用上の注意</b><br>
        当日の混雑状況、開花状況、運行状況、開催可否など、変動する情報は正確性確保のため回答を控える場合があります。
        その場合は、公式サイトや最新のお知らせをご確認ください。
    </div>
    """, unsafe_allow_html=True)

def render_sidebar_help() -> None:
    st.sidebar.markdown("""
    <div class="okz-side-box">
        <div class="okz-side-title">使い方</div>
        <div class="okz-side-text">
            知りたいことを自然文で入力してください。<br>
            例：<br>
            ・スプレー缶は何ごみ？<br>
            ・転入届はどこで出す？<br>
            ・岡崎城の駐車場は？
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
    <div class="okz-side-box">
        <div class="okz-side-title">回答方針</div>
        <div class="okz-side-text">
            公式資料に根拠がある内容のみ回答します。<br>
            根拠が不足する場合は、推測で回答しません。
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
    <div class="okz-side-box">
        <div class="okz-side-title">観光の注意</div>
        <div class="okz-side-text">
            今日・現在・今週など、変動性の高い質問は回答を控える場合があります。
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_admin_footer() -> None:
    st.markdown("""
    <div class="okz-footer">
        <b>免責事項</b><br>
        このサービスは、岡崎市の公開資料等をもとに案内を行うβ版です。制度改正、運用変更、天候、混雑、イベント実施状況などにより、
        実際の情報が異なる場合があります。最終確認は必ず公式サイト・公式資料・各窓口で行ってください。<br><br>

        <b>関連リンク</b><br>
        ・岡崎市公式サイト<br>
        ・岡崎市のごみ・リサイクル関連ページ<br>
        ・岡崎市の観光関連ページ
    </div>
    """, unsafe_allow_html=True)

def render_answer_card(answer_text: str, status: str) -> None:
    label = "回答"
    if status == "abstain":
        label = "回答できません"
    elif status == "ok":
        label = "回答"

    st.markdown(
        f"""
        <div class="okz-answer-card">
            <div class="okz-answer-label">{label}</div>
            <div class="okz-answer-text">{answer_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_cites_block(cites: list[str]) -> None:
    if not cites:
        return

    st.markdown(
        """
        <div class="okz-proof-card">
            <div class="okz-proof-title">根拠・参考URL</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for i, c in enumerate(cites, start=1):
        st.markdown(f"{i}. {c}")
def render_meta_info(category: str, collection_name: str, best_dist: float) -> None:
    dist_text = ""
    try:
        dist_text = f"{float(best_dist):.3f}"
    except Exception:
        dist_text = str(best_dist)

    st.markdown(
        f"""
        <div class="okz-meta-grid">
            <div class="okz-meta-box">
                <div class="okz-meta-label">判定カテゴリ</div>
                <div class="okz-meta-value">{category}</div>
            </div>
            <div class="okz-meta-box">
                <div class="okz-meta-label">使用コレクション</div>
                <div class="okz-meta-value">{collection_name}</div>
            </div>
            <div class="okz-meta-box">
                <div class="okz-meta-label">best_dist</div>
                <div class="okz-meta-value">{dist_text}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
def render_input_intro() -> None:
    st.markdown("""
    <div class="okz-input-wrap">
        <div class="okz-input-title">質問してみましょう</div>
        <div class="okz-input-help">
            岡崎市のごみ、暮らし、観光について、知りたいことをそのまま入力してください。<br>
            例：岡崎城の駐車場は？ / 転入届はどこで出す？ / スプレー缶は何ごみ？
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_loading_hint() -> None:
    st.markdown("""
    <div class="okz-loading-note">
        <b>AIが調べています…</b><br>
        岡崎市の公式資料や登録済みデータを確認しています。少しお待ちください。
    </div>
    """, unsafe_allow_html=True)
def main():
    # ★ st.set_page_config は最初の Streamlit 呼び出しである必要がある
    st.set_page_config(
        page_title="岡崎市AI総合案内 β版",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    ###load_dotenv()
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid.uuid4().hex
    if "request_seq" not in st.session_state:
        st.session_state["request_seq"] = 0

    api_key = OPENAI_API_KEY
    if not api_key:
        st.error("OPENAI_API_KEY が .env にありません。")
        st.stop()

    oa = OpenAI(api_key=api_key)
    waste_client, kanko_client = get_clients()

    #st.title("岡崎市AI総合案内（統合）")
    #st.caption("ごみ＋暮らし＋観光。カテゴリ自動判定し、該当DBのみ検索します。")
    inject_admin_ui_css()
    render_admin_header()
    render_quick_guide()
# sidebar の slider を差し替え
    with st.sidebar:
        render_sidebar_help()
        st.subheader("観光オプション")
        year = st.selectbox("年フィルタ（任意）", [None, 2025, 2026], index=0)

        st.subheader("品質チューニング（dist閾値）")
        th_waste = st.slider("dist閾値：ごみ", 0.60, 1.60, float(DIST_THRESHOLD_BY_CAT["ごみ"]), 0.01)
        th_life  = st.slider("dist閾値：暮らし", 0.60, 1.80, float(DIST_THRESHOLD_BY_CAT["暮らし"]), 0.01)
        th_kanko = st.slider("dist閾値：観光", 0.30, 1.20, float(DIST_THRESHOLD_BY_CAT["観光"]), 0.01)
        st.subheader("テスト情報（任意）")
        suite_default = get_suite_from_url(default="manual")
        suite_name = st.text_input("suite（例: beta_fixed）", value=suite_default)

        run_default = get_run_id_from_url(default="")
        run_id = st.text_input("run_id（例: beta_volatile_20260314_01）", value=run_default)

        st.caption(f"現在のsuite: {suite_name}")
        st.caption(f"現在のrun_id: {run_id or '未指定'}")

    with st.expander("接続チェック（DB/コレクション確認）"):
        st.write("WASTE/LIFE DB:", WASTE_LIFE_DB_DIR)
        st.write("KANKO DB:", KANKO_DB_DIR)
        st.write("WASTE/LIFE collections:", [c.name for c in waste_client.list_collections()])
        st.write("KANKO collections:", [c.name for c in kanko_client.list_collections()])

    mode = st.radio("検索モード", ["自動判定（おすすめ）", "カテゴリ指定"], horizontal=True)
    st.divider()

    render_input_intro()
    st.subheader("よくある質問")

    presets = [
        "指定ごみ袋が必要？種類は？",
        "モバイルバッテリーは何ごみ？",
        "転入したときの手続きは何が必要？",
        "国民健康保険の手続きは？",
        "2026年の夏祭りは？",
        "岡崎城のおすすめの回り方は？",
    ]
    cols = st.columns(3)
    for i, q0 in enumerate(presets):
        if cols[i % 3].button(q0):
            st.session_state["q"] = q0

    with st.form("question_form", clear_on_submit=False):
        q = st.text_input(
            "質問を入力してください",
            key="q",
            placeholder="例：岡崎城と周辺を半日で回るなら？",
            label_visibility="collapsed",
        )

        col1, col2 = st.columns([6, 1])
        with col2:
            st.markdown('<div class="okz-send-btn">', unsafe_allow_html=True)
            submitted = st.form_submit_button("✈ 送信", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    if not submitted and not q:
        st.stop()

    if submitted and not q.strip():
        st.warning("質問を入力してから送信してください。")
        st.stop()

    if not q.strip():
        st.stop()

    # カテゴリ指定モードの場合だけカテゴリ選択を表示
    if mode == "カテゴリ指定":
        fixed_cat = st.selectbox("カテゴリを選択", list(COLLECTIONS.keys()))
        q2 = f"[カテゴリ指定:{fixed_cat}] {q}"
        forced_cat_arg = fixed_cat
    else:
        q2 = q
        forced_cat_arg = None

    DIST_THRESHOLD_BY_CAT["ごみ"] = th_waste
    DIST_THRESHOLD_BY_CAT["暮らし"] = th_life
    DIST_THRESHOLD_BY_CAT["観光"] = th_kanko

    render_loading_hint()

    with st.spinner("AIが回答を作成しています..."):
        ans, cites, ctx, status, chosen_cat, chosen_col, best_dist = answer_portal(
            oa,
            waste_client,
            kanko_client,
            q2,
            year=year,
            forced_cat=forced_cat_arg,
        )

    st.session_state["request_seq"] += 1
    st.session_state["request_id"] = f'{st.session_state["session_id"]}-{st.session_state["request_seq"]:06d}'
    t0 = time.time()
    case_id = st.session_state["request_seq"]   # 1,2,3... と連番になる


    log_path = str(LOG_DIR / "qa_log.jsonl")
    elapsed_ms = int((time.time() - t0) * 1000)

    log_obj = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": st.session_state.get("session_id"),
        "request_id": st.session_state.get("request_id"),

        "category": chosen_cat,
        "query": q,

        "status": status,  # "ok" / "abstain" / "error"
        "answered_by": st.session_state.get("answered_by", "none"),
        "collection_used": st.session_state.get("collection_used"),

        "main_best_dist": st.session_state.get("main_best_dist"),
        "faq_best_dist": st.session_state.get("faq_best_dist"),
        "faq_stage": st.session_state.get("faq_stage"),
        "faq_stage_best": st.session_state.get("faq_stage_best"),
        "faq_query_error": st.session_state.get("faq_query_error"),

        "abstain_hit": st.session_state.get("abstain_hit"),
        "log_main_best_dist": st.session_state.get("main_best_dist"),
        "log_faq_best_dist": st.session_state.get("faq_best_dist"),
        "log_abstain_hit": st.session_state.get("abstain_hit"),
        "dist_threshold_main": DIST_THRESHOLD_BY_CAT.get(chosen_cat),
        "dist_threshold_faq": (FAQ_DIST_THRESHOLD_BY_CAT.get(chosen_cat) if "FAQ_DIST_THRESHOLD_BY_CAT" in globals() else None),



        # ===== 観光：abstain判定の内訳（提出用に必須）=====
        "tourism_dec_reason": st.session_state.get("tourism_dec_reason"),
        "tourism_dec_hit": st.session_state.get("tourism_dec_hit"),
        "tourism_is_fixed_info": st.session_state.get("tourism_is_fixed_info"),
        "top_sources": extract_top_sources(cites, limit=3),
        "answer_preview": (ans or "")[:200],

        "latency_ms": elapsed_ms,

        # 任意：where 条件があるなら残す

        "where": {"year": int(year)} if (chosen_cat == "観光" and year is not None) else None,
    }

    log_obj["suite"] = suite_name
    log_obj["run_id"] = run_id
    log_obj["case_id"] = case_id

    append_log(log_path, log_obj)



    render_meta_info(chosen_cat, chosen_col, best_dist)
    render_answer_card(ans, status)
    #st.subheader("回答")
    #if status == "abstain":
        #st.warning(ans)
    #else:
        #st.write(ans)

    #st.subheader("根拠")
    #st.write("\n".join(cites) if cites else "（根拠を抽出できませんでした）")
    render_cites_block(cites)
   
    if ctx:
        with st.expander("参考資料（抜粋）を見る"):
            st.write(ctx)

    render_admin_footer()
if __name__ == "__main__":
    main()