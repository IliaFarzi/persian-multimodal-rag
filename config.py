"""Environment configuration; relative paths resolve against this repository."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def configured_path(name, default):
    value = Path(os.environ.get(name, default)).expanduser()
    return value if value.is_absolute() else ROOT / value


APP_DATA_DIR = configured_path("APP_DATA_DIR", "app_data")
VGEN_MODEL_PATH = configured_path("VGEN_MODEL_PATH", "models/vgen")
EMBEDDING_MODEL_PATH = configured_path("EMBEDDING_MODEL_PATH", "models/embedding")
ENABLE_DEV_TOOLS = os.environ.get("ENABLE_DEV_TOOLS", "0") == "1"
