from pathlib import Path
import chromadb
from chromadb.config import Settings


def get_client(dbdir: str):
    return chromadb.PersistentClient(
        path=dbdir,
        settings=Settings(anonymized_telemetry=False)
    )


def list_collection_names(client):
    cols = client.list_collections()
    names = []
    for c in cols:
        if hasattr(c, "name"):
            names.append(c.name)
        else:
            names.append(str(c))
    return names


def migrate_collection(src_client, dst_client, collection_name: str, batch_size: int = 200):
    print(f"[START] {collection_name}")
    src = src_client.get_collection(collection_name)
    dst = dst_client.get_or_create_collection(collection_name)

    total = src.count()
    print(f"  count={total}")

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
        print(f"  migrated {offset + len(ids)}/{total}")

    print(f"[DONE] {collection_name} -> dst_count={dst.count()}")


def main():
    portal_root = Path(__file__).resolve().parents[1]
    src_dirs = [
        portal_root / "okazaki_waste_rag" / "chroma_db",
        portal_root / "okazaki_rag" / "chroma_db",
    ]
    dst_dir = portal_root / "chroma_db"

    dst_dir.mkdir(parents=True, exist_ok=True)

    dst_client = get_client(str(dst_dir))

    for src_dir in src_dirs:
        if not src_dir.exists():
            print(f"[SKIP] not found: {src_dir}")
            continue

        print(f"=== SOURCE: {src_dir} ===")
        src_client = get_client(str(src_dir))
        names = list_collection_names(src_client)

        if not names:
            print("  no collections")
            continue

        for name in names:
            migrate_collection(src_client, dst_client, name)

    print("=== FINAL COLLECTIONS IN ROOT ===")
    for name in list_collection_names(dst_client):
        print(" -", name)


if __name__ == "__main__":
    main()