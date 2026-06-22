#!/usr/bin/env bash
set -euo pipefail

# Safe storage cleanup for the RN03-10 DAG-IG run.
#
# Default mode is dry-run. To actually delete:
#   APPLY=1 bash scripts/dagig_train/00_clean_storage_safe.sh
#
# The script is conservative by design. It keeps:
# - the search-test repository
# - RN03-10 zip and extracted data
# - Qwen2.5-VL-7B model caches
# - GroundingDINO repo/checkpoint
# - current run outputs under configurable keep dirs

APPLY="${APPLY:-0}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
REPO_ROOT="${REPO_ROOT:-${DATA_ROOT}/search-test}"
DAGIG_ROOT="${DAGIG_ROOT:-${DATA_ROOT}/dagig}"
DAGIG_TRAIN_ROOT="${DAGIG_TRAIN_ROOT:-${DATA_ROOT}/dagig_train}"
RN_ROOT="${RN_ROOT:-${REPO_ROOT}/data/rn03_10}"
THIRD_PARTY_ROOT="${THIRD_PARTY_ROOT:-${REPO_ROOT}/third_party}"

run() {
  if [[ "$APPLY" == "1" ]]; then
    echo "+ $*"
    "$@"
  else
    echo "[dry-run] $*"
  fi
}

remove_path() {
  local path="$1"
  if [[ -e "$path" || -L "$path" ]]; then
    run rm -rf "$path"
  else
    echo "[skip-missing] $path"
  fi
}

print_header() {
  echo
  echo "========== $1 =========="
}

print_header "Mode"
if [[ "$APPLY" == "1" ]]; then
  echo "APPLY=1: deletion is enabled"
else
  echo "APPLY=0: dry-run only"
fi

print_header "Disk usage"
df -h / /root/autodl-tmp 2>/dev/null || df -h || true

print_header "Large directories under /root/autodl-tmp"
du -h --max-depth=2 "$DATA_ROOT" 2>/dev/null | sort -hr | head -80 || true

print_header "Protected paths"
cat <<EOF
Repository:              $REPO_ROOT
RN03-10 data root:       $RN_ROOT
DAGIG train root:        $DAGIG_TRAIN_ROOT
DAGIG cache root:        $DAGIG_ROOT
Third-party root:        $THIRD_PARTY_ROOT
GroundingDINO expected:  $THIRD_PARTY_ROOT/GroundingDINO
GroundingDINO weights:   $THIRD_PARTY_ROOT/GroundingDINO_weights
EOF

print_header "Known model/cache candidates"
find "$DATA_ROOT" /root/.cache -maxdepth 7 -type d \( \
  -path '*models--Qwen--Qwen2.5-VL-7B-Instruct*' -o \
  -path '*models--Qwen--Qwen2.5-VL-3B-Instruct*' -o \
  -iname '*GroundingDINO*' \
\) 2>/dev/null || true

print_header "Remove safe temporary/runtime caches"
remove_path /tmp/ray
remove_path /tmp/tmp*
remove_path /tmp/pip-*
remove_path /tmp/torch_extensions
remove_path /root/.cache/pip
remove_path /root/.cache/matplotlib
remove_path /root/.cache/torch_extensions

print_header "Remove old DAG-IG/MMSearch temp caches, keep HF model cache"
remove_path "$DAGIG_ROOT/tmp"
remove_path "$DAGIG_ROOT/pip_cache"
remove_path "$DAGIG_ROOT/xdg_cache"
remove_path "$DAGIG_ROOT/matplotlib"

print_header "Remove old FVQA/MMSearch debug artifacts if present"
remove_path "$REPO_ROOT/mmsearch_r1/data/fvqa_debug_images"
remove_path "$REPO_ROOT/mmsearch_r1/data/fvqa_debug_train.pq"
remove_path "$REPO_ROOT/mmsearch_r1/data/fvqa_debug_val.pq"
remove_path "$REPO_ROOT/results/mmsearch_r1"
remove_path "$REPO_ROOT/outputs"
remove_path "$DATA_ROOT/dagig/results_keep/mmsearch_mainline"
remove_path "$DATA_ROOT/dagig/results_keep/mmsearch_mainline_fixed"

print_header "Remove old v1 DAG-IG pilot outputs only if RN03-10 run does not need them"
if [[ "${DELETE_OLD_DAGIG_TRAIN:-0}" == "1" ]]; then
  remove_path "$DAGIG_TRAIN_ROOT/outputs"
  remove_path "$DAGIG_TRAIN_ROOT/results"
  remove_path "$DAGIG_TRAIN_ROOT/data"
else
  echo "Keeping $DAGIG_TRAIN_ROOT/{data,outputs,results}. Set DELETE_OLD_DAGIG_TRAIN=1 to delete them."
fi

print_header "Apt cache cleanup"
if command -v apt-get >/dev/null 2>&1; then
  if [[ "$APPLY" == "1" ]]; then
    apt-get clean || true
    rm -rf /var/lib/apt/lists/* || true
  else
    echo "[dry-run] apt-get clean"
    echo "[dry-run] rm -rf /var/lib/apt/lists/*"
  fi
else
  echo "apt-get not found"
fi

print_header "Python bytecode/cache cleanup inside repo"
if [[ -d "$REPO_ROOT" ]]; then
  if [[ "$APPLY" == "1" ]]; then
    find "$REPO_ROOT" -type d -name "__pycache__" -prune -exec rm -rf {} +
    find "$REPO_ROOT" -type d -name ".pytest_cache" -prune -exec rm -rf {} +
  else
    echo "[dry-run] find $REPO_ROOT -type d -name __pycache__ -delete"
    echo "[dry-run] find $REPO_ROOT -type d -name .pytest_cache -delete"
  fi
fi

print_header "Post-cleanup disk usage"
df -h / /root/autodl-tmp 2>/dev/null || df -h || true
du -h --max-depth=2 "$DATA_ROOT" 2>/dev/null | sort -hr | head -80 || true

print_header "Done"
if [[ "$APPLY" == "1" ]]; then
  echo "Cleanup applied."
else
  echo "Dry-run complete. Re-run with APPLY=1 to delete."
fi
