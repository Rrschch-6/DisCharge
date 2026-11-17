#!/bin/bash
# Generate CodeGuard+ completions (sharded over GPUs), then score them with CodeQL and the unit tests.
#
# Usage:
#   GPUS="0 1" bash codeguard_eval.sh CHECKPOINT OUTPUT_NAME [generate.py flags]   # DisCharge
#   GPUS="0 1" bash codeguard_eval.sh --baseline MODEL_NAME OUTPUT_NAME [generate.py flags]   # unsteered baseline
# Extra flags go to codeguard_ebt.py / CodeGuard+'s generate.py, e.g. --cwes cwe-079,cwe-089 for a subset.
# Environment variables (all optional):
#   GPUS          GPUs to use, one generation process each (default "0"; numbering as in nvidia-smi);
#                 list a GPU twice to run two processes on it
#   CONDA_BASE    use $CONDA_BASE/envs/<env> (default: find the envs by name with `conda run -n`)
#   GEN_ENV       conda env for generation (default discharge);  GEN_PY: path of its python instead
#   EVAL_ENV      conda env for CodeGuard+ scoring (default codeguardplus);  EVAL_PY: path of its python instead
#   CG_DIR        CodeGuard+ checkout (default third_party/CodeGuardPlus)
#   SCORE=0       generate only; score later with codeguard_score.py
#   PRIVATE_TMP=1 run the unit tests with a private /tmp (for shared machines, see README)
#   SEED          generation seed
set -euo pipefail
USAGE="Usage: GPUS=\"0 1\" $0 CHECKPOINT OUTPUT_NAME [flags]   |   $0 --baseline MODEL_NAME OUTPUT_NAME [flags]"
REPO_DIR=$(cd "$(dirname "$0")" && pwd)
if [ "${1:-}" = "--baseline" ]; then
  [ $# -ge 3 ] || { echo "$USAGE"; exit 1; }
  MODEL_ARGS=(--model-name "$2"); NAME=$3; shift 3
else
  [ $# -ge 2 ] || { echo "$USAGE"; exit 1; }
  [ -f "$1" ] || { echo "checkpoint not found: $1"; exit 1; }
  MODEL_ARGS=(--checkpoint "$(realpath "$1")"); NAME=$2; shift 2
fi
CG_DIR=$(realpath "${CG_DIR:-$REPO_DIR/third_party/CodeGuardPlus}")
CONDA=${CONDA_EXE:-$(command -v conda || true)}
env_python() {   # python of conda env $1: $CONDA_BASE/envs/$1 if CONDA_BASE is set, else wherever conda finds it
  if [ -n "${CONDA_BASE:-}" ]; then echo "$CONDA_BASE/envs/$1/bin/python"; return; fi
  local py=""
  [ -n "$CONDA" ] && py=$("$CONDA" run -n "$1" python -c 'import sys; print("PY=" + sys.executable)' 2>/dev/null \
                         | sed -n 's/^PY=//p' || true)
  echo "${py:-<conda env $1 not found>}"
}
GEN_PY=${GEN_PY:-$(env_python "${GEN_ENV:-discharge}")}
EVAL_PY=${EVAL_PY:-$(env_python "${EVAL_ENV:-codeguardplus}")}
read -r -a GPU_LIST <<< "${GPUS:-0}"
N=${#GPU_LIST[@]}
SEED_ARGS=(); [ -n "${SEED:-}" ] && SEED_ARGS=(--seed "$SEED")
LOG_DIR="$REPO_DIR/logs/codeguard_$NAME"

[ -f "$CG_DIR/generate.py" ] || { echo "CodeGuard+ not found at $CG_DIR (run scripts/setup_codeguardplus.sh or set CG_DIR)"; exit 1; }
for PY in "$GEN_PY" "$EVAL_PY"; do [ -x "$PY" ] || { echo "Missing python: $PY (create the env, or set GEN_PY/EVAL_PY; see README, Setup)"; exit 1; }; done
"$GEN_PY" -c "import torch, transformers, accelerate, scipy, numpy" \
  || { echo "generation env ($GEN_PY) is missing a package: pip install -r requirements.txt (README, Setup)"; exit 1; }
"$EVAL_PY" -c "import flask, numpy, scipy, pandas, tqdm, libcst, torch" \
  || { echo "CodeGuard+ env ($EVAL_PY) is missing a package: pip install -r requirements-codeguardplus.txt (README, Setup)"; exit 1; }
[ -x "$HOME/codeql/codeql" ] || { echo "CodeQL not found in ~/codeql: run scripts/setup_codeguardplus.sh (README, Setup)"; exit 1; }
cd "$CG_DIR"
if [ -e "experiments/$NAME" ]; then echo "experiments/$NAME exists in $CG_DIR; pick a new OUTPUT_NAME"; exit 1; fi
mkdir -p "$LOG_DIR"

echo "== generation on GPUs ${GPU_LIST[*]} (shard logs: $LOG_DIR)"
PIDS=()
trap 'kill "${PIDS[@]}" 2>/dev/null || true' INT TERM
for K in $(seq 0 $((N - 1))); do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU_LIST[$K]} \
    "$GEN_PY" -u "$REPO_DIR/codeguard_ebt.py" "${MODEL_ARGS[@]}" --shard "$K/$N" --output_name "$NAME" \
    --eval_type base --do_sample --num_gen 10 "${SEED_ARGS[@]}" "$@" > "$LOG_DIR/shard$K.log" 2>&1 &
  PIDS+=($!)
done
FAILED=0
for K in "${!PIDS[@]}"; do
  wait "${PIDS[$K]}" || { echo "shard $K failed, see $LOG_DIR/shard$K.log"; FAILED=1; }
done
[ $FAILED -eq 0 ] || exit 1

if [ "${SCORE:-1}" = 0 ]; then echo "== generation done; scoring skipped (SCORE=0)"; exit 0; fi
echo "== scoring"
SCORE_ARGS=(); [ "${PRIVATE_TMP:-0}" = 1 ] && SCORE_ARGS=(--private-tmp)
"$EVAL_PY" "$REPO_DIR/codeguard_score.py" "$NAME" --cg-dir "$CG_DIR" "${SCORE_ARGS[@]}"
