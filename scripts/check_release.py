"""Check the public Git index for accidental weights, data, credentials and large files."""
import re
import subprocess

from config import ROOT


def main():
    git_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True).strip()
    if git_root != str(ROOT):
        raise SystemExit("Initialize the separate release repository before running this check.")
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    errors, total, count = [], 0, 0
    secret = re.compile(r"(?:AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|hf_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")
    for name in filter(None, names):
        path = ROOT / name
        count += 1
        if any(part in {"models", "app_data", "sample_data", ".idea", ".venv", ".aws", ".codex", ".agents", "__pycache__", "evaluation_runs"} for part in path.relative_to(ROOT).parts):
            errors.append(f"Private/runtime path tracked: {name}")
        if (path.name.startswith(".env") and path.name != ".env.example") or name == ".streamlit/secrets.toml" or path.suffix in {".pdf", ".bin", ".safetensors", ".sqlite3", ".pyc"}:
            errors.append(f"Private/runtime file tracked: {name}")
        if not path.is_file():
            errors.append(f"Tracked file missing: {name}")
            continue
        size = path.stat().st_size
        total += size
        if size > 10 * 1024**2:
            errors.append(f"File exceeds the 10 MiB portfolio budget: {name}")
        if size < 10 * 1024**2 and secret.search(path.read_bytes().decode("utf-8", errors="ignore")):
            errors.append(f"Possible credential in {name}; inspect locally")
    if not count:
        errors.append("Git index is empty")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Release index passed: {count} files, {total / 1024:.1f} KiB. No weights, runtime data, or recognized credential patterns tracked.")


if __name__ == "__main__":
    main()
