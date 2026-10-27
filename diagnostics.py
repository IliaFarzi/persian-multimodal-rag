"""Cheap setup checks: never load model weights or create application data."""
import json
import shutil
import subprocess
from pathlib import Path

from config import EMBEDDING_MODEL_PATH, VGEN_MODEL_PATH


def model_files_present(path, vision=False):
    path = Path(path)
    required = ["config.json", "tokenizer_config.json"]
    if vision:
        required += ["preprocessor_config.json"]
    if not all((path / name).is_file() for name in required):
        return False
    if not any((path / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "sentencepiece.bpe.model")):
        return False
    if vision and not any((path / name).is_file() for name in ("chat_template.json", "chat_template.jinja")):
        try:
            if not json.loads((path / "tokenizer_config.json").read_text()).get("chat_template"):
                return False
        except (OSError, ValueError, AttributeError):
            return False
    for index in path.glob("*.safetensors.index.json"):
        try:
            shards = set(json.loads(index.read_text())["weight_map"].values())
            return bool(shards) and all((path / shard).is_file() and (path / shard).stat().st_size for shard in shards)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return False
    return any(file.stat().st_size for pattern in ("*.safetensors", "pytorch_model.bin") for file in path.glob(pattern))


def setup_issues(operation="ingestion", include_visual_captions=True):
    issues = []
    if operation == "ingestion":
        if not shutil.which("pdftoppm") or not shutil.which("pdfinfo"):
            issues.append("Install Poppler: sudo apt-get install poppler-utils")
        if not shutil.which("tesseract"):
            issues.append("Install Tesseract and language data: sudo apt-get install tesseract-ocr tesseract-ocr-fas tesseract-ocr-eng")
        else:
            try:
                result = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True, timeout=5, check=True)
                if not {"fas", "eng"}.issubset(set(result.stdout.split())):
                    issues.append("Install Tesseract language data: tesseract-ocr-fas and tesseract-ocr-eng")
            except (OSError, subprocess.SubprocessError):
                issues.append("Tesseract could not list its languages; check your installation.")
    if operation == "qa" or (operation == "ingestion" and include_visual_captions):
        import torch
        if not torch.cuda.is_available():
            issues.append("Gemma FP4 requires an NVIDIA CUDA GPU. Check nvidia-smi and install the CUDA-enabled PyTorch build from README.md.")
        if not model_files_present(VGEN_MODEL_PATH, vision=True):
            issues.append(f"Gemma files are missing at {VGEN_MODEL_PATH}. Run python -m scripts.download_models --model vgen after accepting the upstream license.")
    if operation == "semantic" and not model_files_present(EMBEDDING_MODEL_PATH):
        issues.append(f"E5 files are missing at {EMBEDDING_MODEL_PATH}. Run python -m scripts.download_models --model embedding.")
    return issues


if __name__ == "__main__":
    issues = setup_issues() + setup_issues("semantic")
    print("\n".join(dict.fromkeys(issues)) if issues else "Local dependencies and model files found. Run the GPU smoke check before release.")
    raise SystemExit(bool(issues))
