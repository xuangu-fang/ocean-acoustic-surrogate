#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact_root="${OCEAN_SURROGATE_ROOT:-/mnt/data/xuangu-fang/ocean-acoustics/projects/ocean-acoustic-surrogate}"

cd "${project_root}"
OCEAN_SURROGATE_ROOT="${artifact_root}" uv run python scripts/generate_realistic_report_figures.py

cd docs/technical_report
latexmk -xelatex -interaction=nonstopmode -halt-on-error technical_report_v1.5.tex
cp technical_report_v1.5.pdf ../Ocean_Acoustic_Surrogate_Technical_Report_v1.5.pdf
latexmk -C technical_report_v1.5.tex

echo "${project_root}/docs/Ocean_Acoustic_Surrogate_Technical_Report_v1.5.pdf"
