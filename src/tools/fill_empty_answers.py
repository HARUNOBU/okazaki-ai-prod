import json
from pathlib import Path

SRC = Path("out/faq_drafts.json")
DST = Path("out/faq_drafts_filled.json")

REFUSAL = (
    "資料に明確な記載が見当たらないため、この質問には回答できません。"
    "リンク先PDFの該当箇所をご確認ください。"
)

d = json.loads(SRC.read_text(encoding="utf-8"))
changed = 0
for x in d:
    a = (x.get("answer") or "").strip()
    if not a:
        x["answer"] = REFUSAL
        changed += 1

DST.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"written: {DST} (filled={changed}/{len(d)})")
