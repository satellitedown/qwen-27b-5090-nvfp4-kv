#!/usr/bin/env python3
"""Download the original authors' checkpoints at the measured revisions."""
import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "runtime-manifest.json").read_text())
    for role, model in manifest["models"].items():
        destination = args.models_dir / model["directory"]
        print(f"Downloading {role}: {model['repo_id']} @ {model['revision']}", flush=True)
        snapshot_download(
            repo_id=model["repo_id"],
            revision=model["revision"],
            local_dir=destination,
            allow_patterns=[
                "*.json", "*.safetensors", "*.txt", "*.jinja", "*.model",
                "README.md", "LICENSE*", "NOTICE*", "SHA256SUMS",
            ],
        )
        print(f"Ready: {destination}", flush=True)


if __name__ == "__main__":
    main()
