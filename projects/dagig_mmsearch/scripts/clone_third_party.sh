#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
mkdir -p third_party

clone_if_missing() {
  local url="$1"
  local dir="$2"
  if [ -d "$dir/.git" ]; then
    echo "exists: $dir"
  else
    git clone --depth 1 "$url" "$dir" || echo "clone failed for $url; continue with local scaffold"
  fi
}

clone_if_missing https://github.com/MaYufei-NPU/IG-Search third_party/IG-Search
clone_if_missing https://github.com/EvolvingLMMs-Lab/multimodal-search-r1 third_party/multimodal-search-r1
clone_if_missing https://github.com/mmsearch-plus/MMSearch-Plus third_party/MMSearch-Plus
clone_if_missing https://github.com/EvolvingLMMs-Lab/lmms-eval third_party/lmms-eval
