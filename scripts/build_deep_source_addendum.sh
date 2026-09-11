#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact_root="${OCEAN_SURROGATE_ROOT:-/mnt/data/xuangu-fang/ocean-acoustics/projects/ocean-acoustic-surrogate}"

cd "${project_root}"
OCEAN_SURROGATE_ROOT="${artifact_root}" uv run python scripts/generate_unseen_depth_figures.py "$@"

cd docs/deep_source_addendum
latexmk -xelatex -interaction=nonstopmode -halt-on-error deep_source_addendum.tex
cp deep_source_addendum.pdf ../Deep_Source_Generalization_Addendum_v1.0.pdf
latexmk -C deep_source_addendum.tex

echo "${project_root}/docs/Deep_Source_Generalization_Addendum_v1.0.pdf"
