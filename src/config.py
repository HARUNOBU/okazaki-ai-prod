import os
from pathlib import Path
from dotenv import load_dotenv

ENV_PATH = os.getenv("APP_ENV_FILE", "")

if ENV_PATH and Path(ENV_PATH).exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

APP_ENV = os.getenv("APP_ENV", "local")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

WASTE_LIFE_DB_DIR = Path(os.getenv("WASTE_LIFE_DB_DIR", "./chroma_db"))
KANKO_DB_DIR = Path(os.getenv("KANKO_DB_DIR", "./chroma_db"))
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))

STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
