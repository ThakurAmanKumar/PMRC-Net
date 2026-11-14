from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
V11_ROOT = ROOT / "v1.1"
MANIFEST_PATH = V11_ROOT / "reports" / "v1_0_preservation_manifest.json"
EXCLUDED_DIRECTORIES = {".git", ".venv", "__pycache__", ".pytest_cache"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def historical_files() -> dict[str, str]:
    files: dict[str, str] = {}
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_DIRECTORIES for part in relative.parts):
            continue
        if relative.parts[0] == "v1.1" or not path.is_file():
            continue
        files[relative.as_posix()] = sha256_file(path)
    return dict(sorted(files.items()))


def write_baseline() -> dict[str, Any]:
    files = historical_files()
    payload = {
        "baseline": "v1.0 workspace before v1.1 implementation",
        "root": str(ROOT),
        "excluded_directories": sorted(EXCLUDED_DIRECTORIES | {"v1.1"}),
        "file_count": len(files),
        "files_sha256": files,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def verify_baseline() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Preservation manifest not found: {MANIFEST_PATH}")
    baseline = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = baseline["files_sha256"]
    current = historical_files()
    expected_paths = set(expected)
    current_paths = set(current)
    changed = sorted(path for path in expected_paths & current_paths if expected[path] != current[path])
    removed = sorted(expected_paths - current_paths)
    added = sorted(current_paths - expected_paths)
    return {
        "status": "PASS" if not (changed or removed or added) else "FAIL",
        "baseline_file_count": len(expected),
        "current_file_count": len(current),
        "changed": changed,
        "removed": removed,
        "added": added,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = verify_baseline() if args.verify else write_baseline()
    print(json.dumps(result, indent=2))
    if args.verify and result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()