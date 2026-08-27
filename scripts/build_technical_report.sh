#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact_root="${OCEAN_SURROGATE_ROOT:-/home/ubuntu/ocean-acoustic-surrogate-artifacts}"

cd "${project_root}"
OCEAN_SURROGATE_ROOT="${artifact_root}" uv run python scripts/generate_technical_report_figures.py

cd docs/technical_report
latexmk -xelatex -interaction=nonstopmode -halt-on-error technical_report.tex
cp technical_report.pdf ../Ocean_Acoustic_Surrogate_Technical_Report_v1.1.pdf
latexmk -C technical_report.tex

echo "${project_root}/docs/Ocean_Acoustic_Surrogate_Technical_Report_v1.1.pdf"
