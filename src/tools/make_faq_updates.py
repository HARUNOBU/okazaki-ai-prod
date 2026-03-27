import json
from pathlib import Path

EXISTING = Path(r"..\data\waste\faq_waste.jsonl")
DRAFTS   = Path(r"out\faq_drafts_waste_clean.json")
OUT      = Path(r"out\faq_waste_updates.jsonl")

def norm(s: str) -> str:
    return (s or "").strip().replace("　", " ").lower()

def read_jsonl(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows

def main():
    existing = read_jsonl(EXISTING)
    drafts = json.loads(DRAFTS.read_text(encoding="utf-8"))

    # question -> existing row
    ex_by_q = {norm(r.get("question","")): r for r in existing}

    updates = []
    misses = []
    for d in drafts:
        qn = norm(d.get("question",""))
        ex = ex_by_q.get(qn)
        if not ex:
            misses.append(d.get("question",""))
            continue

        # 既存qidを引き継いで更新
        u = dict(ex)
        # drafts側で上書きしたいものだけ更新
        u["answer"] = d.get("answer", u.get("answer",""))
        u["aliases"] = d.get("aliases", u.get("aliases", []))
        u["pdf_url"] = d.get("pdf_url", u.get("pdf_url",""))
        u["page"] = d.get("page", u.get("page",""))
        updates.append(u)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in updates) + ("\n" if updates else ""), encoding="utf-8")

    print(f"existing={len(existing)} drafts={len(drafts)} updates={len(updates)} misses={len(misses)}")
    if misses:
        print("misses (not found in existing by question):")
        for m in misses:
            print(" -", m)
    print("written:", OUT)

if __name__ == "__main__":
    main()
