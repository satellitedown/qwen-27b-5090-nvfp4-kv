#!/usr/bin/env python3
"""Create a hash-checked SGLang overlay without modifying site-packages."""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_hashes(root, expected):
    for relative, digest in expected.items():
        path = root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if actual != digest:
            raise RuntimeError(f"Source mismatch: {path}; expected {digest}, got {actual}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runtime")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "runtime-manifest.json").read_text())
    package = importlib.metadata.distribution("sglang")
    if package.version != manifest["base_versions"]["sglang"]:
        raise RuntimeError(f"Expected SGLang 0.5.20; found {package.version}")
    source = Path(package.locate_file(""))
    check_hashes(source, manifest["base_source_sha256"])
    patch = ROOT / manifest["patch"]
    if hashlib.sha256(patch.read_bytes()).hexdigest() != manifest["patch_sha256"]:
        raise RuntimeError("Patch checksum does not match runtime-manifest.json")
    output = args.output.resolve()
    if output.exists():
        check_hashes(output, manifest["patched_source_sha256"])
        print(f"Already patched and verified: {output}")
        return
    if not shutil.which("patch"):
        raise RuntimeError("GNU patch is required; install it with your system package manager")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sglang-patch-", dir=output.parent) as temp:
        staging = Path(temp) / "runtime"
        shutil.copytree(
            source / "sglang", staging / "sglang",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        subprocess.run(
            ["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(patch)],
            cwd=staging, check=True,
        )
        check_hashes(staging, manifest["patched_source_sha256"])
        staging.rename(output)
    print(f"Patched and verified: {output}")


if __name__ == "__main__":
    main()
