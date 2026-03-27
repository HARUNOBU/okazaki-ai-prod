from pathlib import Path
import chromadb
from chromadb.config import Settings

portal_root = Path(__file__).resolve().parents[1]
dbdir = portal_root / "chroma_db"

client = chromadb.PersistentClient(
    path=str(dbdir),
    settings=Settings(anonymized_telemetry=False)
)

cols = client.list_collections()
for c in cols:
    name = c.name if hasattr(c, "name") else str(c)
    print(name)