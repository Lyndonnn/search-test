from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Hugging Face model access before a full download.")
    parser.add_argument("--repo-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--filename", default="tokenizer_config.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"repo_id={args.repo_id}")
    print(f"filename={args.filename}")
    print(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '<default huggingface.co>')}")
    print(f"HF_HOME={os.environ.get('HF_HOME', '')}")
    print(f"HF_HUB_ETAG_TIMEOUT={os.environ.get('HF_HUB_ETAG_TIMEOUT', '')}")
    print(f"HF_HUB_DOWNLOAD_TIMEOUT={os.environ.get('HF_HUB_DOWNLOAD_TIMEOUT', '')}")
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            cache_dir=os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HOME"),
        )
    except Exception as exc:
        raise SystemExit(f"HF probe failed: {exc}") from exc
    print(f"HF probe ok: {path}")


if __name__ == "__main__":
    main()
