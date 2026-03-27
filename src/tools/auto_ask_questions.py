import json
import os
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# あなたの本番コードを再利用
import app_portal_v2_patched as ap  # app_portal_v2.py

# 質問データ（場所は必要に応じて変更）
QUESTIONS_JSON = ROOT / "data" / "waste" / "sample_questions_waste_100.json"

# ログを増やすときの安全設定
SLEEP_SEC = 1.2           # API叩きすぎ防止
LIMIT = None              # 例: 30 にすると最初の30問だけ


def init_runtime():
    """Streamlitを起動せずに、本番と同等の runtime を作る"""
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が .env にありません。")

    oa = OpenAI(api_key=api_key)

    # app_portal_v2 にある get_clients() を使う
    waste_client, kanko_client = ap.get_clients()

    return oa, waste_client, kanko_client

def main():
    oa, waste_client, kanko_client = init_runtime()

    if not QUESTIONS_JSON.exists():
        raise FileNotFoundError(f"Questions JSON not found: {QUESTIONS_JSON}")

    with open(QUESTIONS_JSON, encoding="utf-8") as f:
        questions = json.load(f)
    if not isinstance(questions, list):
        raise ValueError("Questions JSON must be a list")

    # Streamlitの session_state に近い概念を auto_ask 側でも持つ
    session_id = uuid.uuid4().hex
    request_seq = 0

    # ログ出力先（あなたの Streamlit と同じ場所に寄せる）
    # app_portal_v2 の PORTAL_ROOT があるならそれを優先
    portal_root = getattr(ap, "PORTAL_ROOT", ROOT)
    log_path = str(Path(portal_root) / "logs" / "qa_log.jsonl")
    os.makedirs(Path(log_path).parent, exist_ok=True)

    n_total = len(questions)
    n_run = min(n_total, LIMIT) if isinstance(LIMIT, int) else n_total
    print(f"[INFO] loaded questions: {n_total}, run: {n_run}")
    print(f"[INFO] json: {QUESTIONS_JSON}")
    print(f"[INFO] log : {log_path}")

    for i, item in enumerate(questions[:n_run], start=1):
        q = (item.get("question") or "").strip()
        if not q:
            continue

        request_seq += 1
        t0 = time.time()

        print(f"\n[{i:03d}/{n_run:03d}] [ASK] {q}")

        ans, cites, ctx, status, chosen_cat, chosen_col, best_dist = ap.answer_portal(
            oa,
            waste_client,
            kanko_client,
            q,
            year=None,
            forced_cat="waste",
        )

        latency_ms = int((time.time() - t0) * 1000)

        # answer_portal の内部で st.session_state に書いている値があれば拾う
        st = getattr(ap, "st", None)
        ss = getattr(st, "session_state", {}) if st is not None else {}

        main_best_dist = ss.get("main_best_dist")
        faq_best_dist = ss.get("faq_best_dist")
        collection_used = ss.get("collection_used")
        answered_by = ss.get("answered_by", "main")

        # thresholds は app_portal_v2 の定数があればそれをそのまま入れる
        thresholds = None
        if hasattr(ap, "DIST_THRESHOLD_BY_CAT"):
            try:
                thresholds = dict(ap.DIST_THRESHOLD_BY_CAT)
            except Exception:
                thresholds = None

        ap.append_log(log_path, {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "session_id": session_id,
            "request_seq": request_seq,
            "query": q,
            "mode": "auto_ask",
            "forced_cat": "waste",
            "chosen_cat": chosen_cat,
            "collection": chosen_col,
            "collection_used": collection_used,
            "status": status,
            "best_dist": round(float(best_dist), 4) if best_dist is not None else None,
            "main_best_dist": main_best_dist,
            "faq_best_dist": faq_best_dist,
            "answered_by": answered_by,
            "latency_ms": latency_ms,
            "num_cites": len(cites) if isinstance(cites, list) else 0,
            "cites": (cites[:10] if isinstance(cites, list) else []),
            "thresholds": thresholds,
        })

        print(f" -> status={status}, cat={chosen_cat}, col={chosen_col}, best_dist={best_dist}, answered_by={answered_by}, latency_ms={latency_ms}")
        time.sleep(SLEEP_SEC)

    print("\n[INFO] done.")


if __name__ == "__main__":
    main()