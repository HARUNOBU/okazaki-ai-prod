import json, re
from pathlib import Path

SRC = Path("out/faq_drafts_filled.json")
DST = Path("out/faq_drafts_waste_clean.json")

REFUSAL = (
    "資料に明確な記載が見当たらないため、この質問には回答できません。"
    "リンク先PDFの該当箇所をご確認ください。"
)

def clean_pdf_url(u: str) -> str:
    u = (u or "").strip()
    # "（p.5）" 等が混入していたら除去
    u = re.split(r"[（(]", u, maxsplit=1)[0].strip()
    return u

def main():
    rows = json.loads(SRC.read_text(encoding="utf-8"))
    out = []
    for r in rows:
        if (r.get("category") or "").strip() != "ごみ":
            continue

        r2 = dict(r)
        r2["pdf_url"] = clean_pdf_url(r2.get("pdf_url", ""))

        # answer が空なら拒否文で埋める（allow-empty-answer=false 対策）
        if not (r2.get("answer") or "").strip():
            r2["answer"] = REFUSAL

        out.append(r2)

    DST.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written: {DST} (rows={len(out)})")
    print("pdf_url:", [x.get("pdf_url") for x in out])
    print("answer_empty:", sum(1 for x in out if not (x.get('answer') or '').strip()))

if __name__ == "__main__":
    main()
