from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = Path("/mnt/workspace/models/bge-small-zh-v1.5")
DB_DIR = ROOT / "data" / "chroma"
COLLECTION_NAME = "howtocook_recipes"