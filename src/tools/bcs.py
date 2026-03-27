import csv

with open("out/abstain_all.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# idx >= 39 のみ抽出
normal60_abst = [r for r in rows if int(r["idx"]) >= 39]

print("件数:", len(normal60_abst))
for r in normal60_abst:
    print("-", r["query"])
