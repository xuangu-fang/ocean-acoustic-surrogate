#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
agent_root="${OCEAN_AGENT_ROOT:-$(dirname "${project_root}")/ocean-acoustic-agent}"
artifact_root="${OCEAN_SURROGATE_ROOT:-/mnt/data/xuangu-fang/ocean-acoustics/projects/ocean-acoustic-surrogate}"
mode="${REPRO_MODE:-check}"
samples="${REPRO_SAMPLES:-256}"
run_dir="${REPRO_RUN_DIR:-}"
config="${REPRO_CONFIG:-configs/realistic_terrain_mvp.yaml}"
campaign="${REPRO_CAMPAIGN:-configs/realistic_campaign.yaml}"
experiment="${REPRO_EXPERIMENT:-real_r4_terrain_anchor_large_fno}"

export OCEAN_SURROGATE_ROOT="${artifact_root}"

usage() {
  printf '%s\n' \
    "Usage: REPRO_MODE=<check|train|full|verify> bash scripts/reproduce_mvp.sh" \
    "" \
    "  check   Sync dependencies and run project tests; no Bellhop labels are generated." \
    "  train   Train and verify the final model from an existing frozen dataset." \
    "  full    Check Bellhop, run convergence, generate/resume labels, train, and verify." \
    "  verify  Verify REPRO_RUN_DIR on the sealed test split without retraining." \
    "" \
    "Optional variables:" \
    "  OCEAN_AGENT_ROOT       sibling ocean-acoustic-agent repository" \
    "  OCEAN_SURROGATE_ROOT   large artifact root" \
    "  REPRO_CONFIG           task/data config, default configs/realistic_terrain_mvp.yaml" \
    "  REPRO_CAMPAIGN         experiment config, default configs/realistic_campaign.yaml" \
    "  REPRO_EXPERIMENT       experiment id to train" \
    "  REPRO_SAMPLES          sample count, default 256" \
    "  REPRO_RUN_DIR          run directory required by verify mode"
}

if [[ "${mode}" == "help" || "${mode}" == "--help" || "${mode}" == "-h" ]]; then
  usage
  exit 0
fi

case "${mode}" in
  check|train|full|verify) ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if [[ ! -d "${agent_root}" ]]; then
  printf 'Missing ocean-acoustic-agent sibling repository: %s\n' "${agent_root}" >&2
  exit 1
fi

cd "${project_root}"
uv python install 3.11
uv sync --locked
uv run ruff check .
uv run pytest -q

if [[ "${mode}" == "check" ]]; then
  printf 'Project check passed. Bellhop was not executed.\n'
  exit 0
fi

dataset_path="$(uv run python - "${config}" "${samples}" <<'PY'
import sys
from ocean_acoustic_surrogate.config import MVPConfig

config = MVPConfig.from_yaml(sys.argv[1])
print(config.dataset_root / f"n{sys.argv[2]}" / "dataset.npz")
PY
)"

if [[ "${mode}" == "full" ]]; then
  (
    cd "${agent_root}"
    uv python install 3.11
    uv sync --locked --extra dev
    uv run python scripts/check_env.py
  )
  uv run ocean-acoustic-surrogate pilot "${config}" --samples 8
  uv run ocean-acoustic-surrogate generate "${config}" --samples "${samples}"
  uv run ocean-acoustic-surrogate profile "${config}" --samples "${samples}"
fi

if [[ "${mode}" == "train" || "${mode}" == "full" ]]; then
  if [[ ! -f "${dataset_path}" ]]; then
    printf 'Frozen dataset not found: %s\n' "${dataset_path}" >&2
    printf 'Copy the sealed dataset there, or rerun with REPRO_MODE=full.\n' >&2
    exit 1
  fi
  uv run ocean-acoustic-surrogate campaign \
    "${config}" "${campaign}" \
    --samples "${samples}" --only "${experiment}"
  run_dir="$(uv run python -c '
import json, os
from pathlib import Path
root = Path(os.environ["OCEAN_SURROGATE_ROOT"])
summary = json.loads((root / "campaigns/latest.json").read_text())
print(summary["best"]["run_dir"])
')"
fi

if [[ "${mode}" == "verify" && -z "${run_dir}" ]]; then
  printf 'REPRO_RUN_DIR is required when REPRO_MODE=verify.\n' >&2
  exit 1
fi

uv run ocean-acoustic-surrogate verify \
  "${config}" "${run_dir}" --samples "${samples}" --device auto

printf 'Reproduction completed.\nDataset: %s\nRun: %s\n' "${dataset_path}" "${run_dir}"
