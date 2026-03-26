from pathlib import Path
import sys
import chromadb
from chromadb.config import Settings

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import WASTE_LIFE_DB_DIR

client = chromadb.PersistentClient(
    path=str(WASTE_LIFE_DB_DIR),
    settings=Settings(anonymized_telemetry=False)
)

cols = client.list_collections()
for c in cols:
    name = c.name if hasattr(c, "name") else str(c)
    print(name)
