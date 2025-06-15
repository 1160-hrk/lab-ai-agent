"""環境変数とパス設定"""
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parents[1]
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHROMA_PERSIST_PATH = Path(os.getenv("CHROMA_PERSIST_PATH", BASE_DIR / "data" / "chroma"))
