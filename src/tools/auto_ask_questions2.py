import json
import time
import sys
from pathlib import Path

# src を import パスに追加（超重要）
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app_portal_v2_patched import answer_portal  # ← ★ここを修正

QUESTIONS_JSON = "data/waste/sample_questions_waste_100.json"

def main():
    with open(QUESTIONS_JSON, encoding="utf-8") as f:
        questions = json.load(f)

    for q in questions:
        text = q["question"]
        print(f"[ASK] {text}")

        ans, cites, ctx, status, chosen_cat, chosen_col, best_dist = answer_portal(
            q=text,
            year=None,
            forced_cat="waste",
        )

        print(f" -> status={status}, cat={chosen_cat}, dist={best_dist}")
        time.sleep(1.0)

if __name__ == "__main__":
    main()

    with open(QUESTIONS_JSON, encoding="utf-8") as f:
        questions = json.load(f)

    for q in questions:
        text = q["question"]
        print(f"[ASK] {text}")

        ans, cites, ctx, status, chosen_cat, chosen_col, best_dist = answer_portal(
            q=text,
            year=None,
            forced_cat="waste"
        )

        print(f" -> status={status}, cat={chosen_cat}, dist={best_dist}")
        time.sleep(1.0)  # ★API叩きすぎ防止

if __name__ == "__main__":
    main()
