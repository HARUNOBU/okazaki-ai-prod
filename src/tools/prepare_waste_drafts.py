import json
import re
from pathlib import Path

SRC = Path("out/faq_drafts_filled.json")
DST = Path("out/faq_drafts_waste_clean.json")

def clean_pdf_url(u: str) -> str:
    u = (u or "").strip()
    # 「（p.5）」のような注記が混ざっている場合は除去
    u = re.split(r"[（(]", u, maxsplit=1)[0].strip()
    return u

def to_int_or_keep(x):
    s = (x or "").strip()
    if s.isdigit():
        return int(s)
    return s

def main():
    rows = json.loads(SRC.read_text(encoding="utf-8"))
    out = []
    for r in rows:
        if (r.get("category") or "").strip() != "ごみ":
            continue

        r2 = dict(r)
        r2["pdf_url"] = clean_pdf_url(r2.get("pdf_url", ""))
        r2["page"] = to_int_or_keep(r2.get("page", ""))
        out.append(r2)

    DST.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written: {DST} (rows={len(out)})")

if __name__ == "__main__":
    main()
