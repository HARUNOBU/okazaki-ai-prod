from pathlib import Path
import sys
import chromadb
from chromadb.config import Settings

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import WASTE_LIFE_DB_DIR, KANKO_DB_DIR

ROOT_DB = Path(WASTE_LIFE_DB_DIR)

SOURCES = {
    "okazaki_waste": Path(WASTE_LIFE_DB_DIR),
    "okazaki_waste_faq": Path(WASTE_LIFE_DB_DIR),
    "okazaki_life": Path(WASTE_LIFE_DB_DIR),
    "okazaki_events": Path(KANKO_DB_DIR),
    "okazaki_events_faq": Path(KANKO_DB_DIR),
    "okazaki_events_fixed_faq": Path(KANKO_DB_DIR),
}

TARGET_COLLECTIONS = [
    "okazaki_waste",
    "okazaki_waste_faq",
    "okazaki_life",
    "okazaki_events",
    "okazaki_events_faq",
    "okazaki_events_fixed_faq",
]

def get_client(dbdir: Path):
    return chromadb.PersistentClient(
        path=str(dbdir),
        settings=Settings(anonymized_telemetry=False)
    )

def copy_collection(src_client, dst_client, collection_name: str, batch_size: int = 200):
    src = src_client.get_collection(collection_name)
    dst = dst_client.get_or_create_collection(collection_name)

    total = src.count()
    print(f"[COPY] {collection_name}: total={total}")

    for offset in range(0, total, batch_size):
        got = src.get(
            include=["documents", "metadatas", "embeddings"],
            limit=min(batch_size, total - offset),
            offset=offset,
        )

        ids = got.get("ids", [])
        docs = got.get("documents", [])
        metas = got.get("metadatas", [])
        embs = got.get("embeddings", [])

        if not ids:
            continue

        dst.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embs,
        )
        print(f"  -> {offset + len(ids)}/{total}")

    print(f"[DONE] {collection_name}: dst_count={dst.count()}")

def main():
    ROOT_DB.mkdir(parents=True, exist_ok=True)
    dst_client = get_client(ROOT_DB)

    for name in TARGET_COLLECTIONS:
        src_dir = SOURCES[name]
        print("=" * 80)
        print(f"collection={name}")
        print(f"source={src_dir}")
        print(f"target={ROOT_DB}")

        if not src_dir.exists():
            print("  source db not found, skip")
            continue

        src_client = get_client(src_dir)

        names = []
        for c in src_client.list_collections():
            names.append(c.name if hasattr(c, "name") else str(c))

        if name not in names:
            print("  collection not found in source, skip")
            continue

        copy_collection(src_client, dst_client, name)

    print("=" * 80)
    print("[FINAL]")
    for c in dst_client.list_collections():
        cname = c.name if hasattr(c, "name") else str(c)
        col = dst_client.get_collection(cname)
        print(f"{cname}: {col.count()}")

if __name__ == "__main__":
    main()
