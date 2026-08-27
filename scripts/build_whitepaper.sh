#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact_root="${OCEAN_SURROGATE_ROOT:-/home/ubuntu/ocean-acoustic-surrogate-artifacts}"

cd "${project_root}"
OCEAN_SURROGATE_ROOT="${artifact_root}" uv run python scripts/generate_whitepaper_figures.py

cd docs/whitepaper
latexmk -xelatex -interaction=nonstopmode -halt-on-error whitepaper.tex
cp whitepaper.pdf ../Ocean_Acoustic_Surrogate_Whitepaper_v1.0.pdf
latexmk -C whitepaper.tex

echo "${project_root}/docs/Ocean_Acoustic_Surrogate_Whitepaper_v1.0.pdf"
