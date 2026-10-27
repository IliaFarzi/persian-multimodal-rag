"""Download upstream model artifacts at pinned revisions; never save credentials."""
import argparse
import json

from config import ROOT, VGEN_MODEL_PATH, EMBEDDING_MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all", "vgen", "embedding"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "models.lock.json").read_text())
    paths = {"vgen": VGEN_MODEL_PATH, "embedding": EMBEDDING_MODEL_PATH}
    for name in paths if args.model == "all" else [args.model]:
        spec = manifest[name]
        print(f"{spec['repo_id']}@{spec['revision']} -> {paths[name]}")
        print(f"Upstream license: {spec['license']}")
        if args.dry_run:
            continue
        from huggingface_hub import snapshot_download
        try:
            snapshot_download(
                repo_id=spec["repo_id"], revision=spec["revision"], local_dir=paths[name],
                allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "*.txt", "README.md", "1_Pooling/*"],
                ignore_patterns=["onnx/*", "openvino/*"],
            )
        except Exception as exc:
            raise SystemExit(f"Download failed ({type(exc).__name__}). Check connectivity and upstream access. For Gemma, accept its license and authenticate with hf auth login or HF_TOKEN.") from None
        (paths[name] / ".model-revision.json").write_text(json.dumps(spec, indent=2) + "\n")


if __name__ == "__main__":
    main()
