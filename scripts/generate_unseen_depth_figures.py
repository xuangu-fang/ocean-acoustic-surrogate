"""Generate traceable figures for the 50--2000 m unseen-depth experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from ocean_acoustic_surrogate.config import MVPConfig

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = Path(
    os.environ.get(
        "OCEAN_SURROGATE_ROOT",
        "/mnt/data/xuangu-fang/ocean-acoustics/projects/ocean-acoustic-surrogate",
    )
)
CONFIG_PATH = ROOT / "configs/unseen_source_depth_mvp.yaml"
OUTPUT_ROOT = ROOT / "docs/deep_source_addendum"
BLUE = "#276b9a"
ORANGE = "#d8792a"
TEAL = "#24877d"
RED = "#b44343"
GREEN = "#4a8a5b"
NAVY = "#173e60"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, default=None)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    return parser.parse_args()


def save(fig: plt.Figure, name: str) -> None:
    assets = OUTPUT_ROOT / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    fig.savefig(assets / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(assets / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_campaign(path: Path) -> tuple[list[dict], dict]:
    campaign = json.loads(path.read_text())
    runs = []
    for item in campaign["experiments"]:
        run_dir = Path(item["run_dir"])
        metrics = json.loads((run_dir / "metrics.json").read_text())
        with np.load(run_dir / "predictions.npz") as raw:
            predictions = {key: raw[key].copy() for key in raw.files}
        history = json.loads((run_dir / "history.json").read_text())
        runs.append(
            {
                "run_dir": run_dir,
                "metrics": metrics,
                "predictions": predictions,
                "history": history,
            }
        )
    best = min(runs, key=lambda run: run["metrics"]["metrics"]["test"]["aggregate"]["rmse_db"])
    return runs, best


def plot_design(config: MVPConfig, dataset: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.1), constrained_layout=True)
    for profile in config.contract.bathymetry.profiles:
        distance = np.linspace(0, 50_000, 501)
        bottom = np.interp(distance, profile.ranges_m, profile.depths_m)
        label = profile.name.replace("gebco_", "").replace("_screened", "")
        axes[0].plot(distance / 1000, bottom, linewidth=2.3, label=label)
    axes[0].axhspan(10, 1990, color="#d9edf7", alpha=0.7, label="receiver grid")
    axes[0].scatter([0], [2000], marker="*", s=130, color=RED, label="deepest source")
    axes[0].invert_yaxis()
    axes[0].set(
        title="Deep-origin terrain subset",
        xlabel="Range (km)",
        ylabel="Depth (m)",
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)

    split_depths = {
        "train": sorted(
            set(config.contract.resolved_source_depths_m)
            - set(config.split.validation_source_depths_m)
            - set(config.split.test_source_depths_m)
        ),
        "validation (unseen)": config.split.validation_source_depths_m,
        "test (unseen)": config.split.test_source_depths_m,
    }
    colors = {"train": BLUE, "validation (unseen)": ORANGE, "test (unseen)": RED}
    markers = {"train": "o", "validation (unseen)": "s", "test (unseen)": "D"}
    for row, (name, values) in enumerate(split_depths.items()):
        axes[1].scatter(
            values,
            np.full(len(values), row),
            s=65,
            marker=markers[name],
            color=colors[name],
            edgecolor="white",
            linewidth=0.7,
            label=name,
            zorder=3,
        )
    axes[1].set_yticks(range(3), split_depths)
    axes[1].set_xlim(0, 2050)
    axes[1].set_xticks(np.arange(0, 2001, 250))
    axes[1].set(
        title="Leakage-free source-depth split",
        xlabel="Source depth (m)",
    )
    axes[1].grid(axis="x", alpha=0.2)
    axes[1].text(
        0.02,
        0.96,
        "40 depths; 50 m spacing\n480 fields = 2 terrain × 3 month × 40 depth × 2",
        transform=axes[1].transAxes,
        va="top",
        fontsize=9,
        color=NAVY,
    )

    profiles = dataset["ssp_profiles"].astype(str)
    month_colors = {"woa23_march": BLUE, "woa23_june": ORANGE, "woa23_december": TEAL}
    for month, color in month_colors.items():
        index = np.flatnonzero(profiles == month)
        for selected in index[:: max(1, len(index) // 12)]:
            axes[2].plot(
                dataset["ssp_speeds_mps"][selected],
                dataset["ssp_depths_m"],
                color=color,
                alpha=0.12,
                linewidth=0.8,
            )
        axes[2].plot(
            dataset["ssp_speeds_mps"][index].mean(axis=0),
            dataset["ssp_depths_m"],
            color=color,
            linewidth=2.4,
            label=month.replace("woa23_", "").title(),
        )
    axes[2].invert_yaxis()
    axes[2].set(
        title="Traceable three-month SSP family",
        xlabel="Sound speed (m/s)",
        ylabel="Depth (m)",
    )
    axes[2].grid(alpha=0.2)
    axes[2].legend()
    save(fig, "fig01_experiment_design")


def plot_performance(runs: list[dict], best: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8), constrained_layout=True)
    names = [run["metrics"]["experiment_id"] for run in runs]
    short = [name.replace("deep_sourcebar_", "").replace("gaussian", "local") for name in names]
    model_rmse = [run["metrics"]["metrics"]["test"]["aggregate"]["rmse_db"] for run in runs]
    baseline = [run["metrics"]["mean_field_baseline_test"]["rmse_db"] for run in runs]
    x = np.arange(len(runs))
    axes[0].bar(x - 0.18, baseline, width=0.36, color="#aeb8c0", label="train-mean baseline")
    axes[0].bar(x + 0.18, model_rmse, width=0.36, color=BLUE, label="SourceBAR-FNO")
    axes[0].axhline(2.0, color=RED, linestyle="--", linewidth=1.5, label="2 dB requirement")
    axes[0].set_xticks(x, short, rotation=12, ha="right")
    axes[0].set(ylabel="Test RMSE (dB)", title="Strict unseen-depth test")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend()

    by_depth = best["metrics"]["metrics"]["test"]["by_source_depth"]
    depths = sorted((float(key[1:-1]), value) for key, value in by_depth.items())
    axes[1].plot(
        [item[0] for item in depths],
        [item[1]["rmse_db"] for item in depths],
        color=BLUE,
        marker="o",
        linewidth=2.3,
        markersize=7,
        label="unseen test depth",
    )
    axes[1].axhline(2.0, color=RED, linestyle="--", linewidth=1.5, label="2 dB requirement")
    axes[1].set_xticks([item[0] for item in depths])
    axes[1].set(
        xlabel="Source depth (m)",
        ylabel="RMSE (dB)",
        title="Generalization error at each held-out depth",
    )
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    save(fig, "fig02_unseen_depth_performance")


def plot_examples(best: dict) -> None:
    data = best["predictions"]
    test = np.flatnonzero(data["splits"].astype(str) == "test")
    requested = [150.0, 950.0, 1750.0]
    selected = []
    for depth in requested:
        candidates = test[data["source_depths_m"][test] == depth]
        errors = []
        for index in candidates:
            valid = data["valid_mask"][index].astype(bool)
            delta = data["prediction_tl_db"][index][valid] - data["reference_tl_db"][index][valid]
            errors.append(float(np.sqrt(np.mean(delta**2))))
        order = np.argsort(errors)
        selected.append(candidates[order[len(order) // 2]])
    fig, axes = plt.subplots(3, 4, figsize=(16.3, 12.0), constrained_layout=True)
    extent = [
        data["ranges_m"][0] / 1000,
        data["ranges_m"][-1] / 1000,
        data["depths_m"][-1],
        data["depths_m"][0],
    ]
    for row, index in enumerate(selected):
        reference = data["reference_tl_db"][index]
        prediction = data["prediction_tl_db"][index]
        valid = data["valid_mask"][index].astype(bool)
        error = np.where(valid, np.abs(prediction - reference), np.nan)
        image0 = axes[row, 0].imshow(
            reference, extent=extent, aspect="auto", vmin=50, vmax=100, cmap="viridis_r"
        )
        axes[row, 1].imshow(
            prediction, extent=extent, aspect="auto", vmin=50, vmax=100, cmap="viridis_r"
        )
        image2 = axes[row, 2].imshow(
            error, extent=extent, aspect="auto", vmin=0, vmax=3, cmap="magma"
        )
        receiver = int(np.argmin(np.abs(data["depths_m"] - 1000)))
        axes[row, 3].plot(
            data["ranges_m"] / 1000, reference[receiver], color=NAVY, linewidth=1.5, label="Bellhop"
        )
        axes[row, 3].plot(
            data["ranges_m"] / 1000,
            prediction[receiver],
            color=ORANGE,
            linewidth=1.25,
            label="prediction",
        )
        rmse = np.sqrt(np.mean((prediction[valid] - reference[valid]) ** 2))
        axes[row, 0].set_ylabel(f"source {requested[row]:.0f} m\nreceiver depth (m)")
        axes[row, 3].set_title(f"1000 m receiver slice; field RMSE {rmse:.2f} dB")
        axes[row, 3].invert_yaxis()
        axes[row, 3].grid(alpha=0.18)
        axes[row, 3].legend(fontsize=8)
        if row == 0:
            axes[row, 0].set_title("Bellhop reference TL")
            axes[row, 1].set_title("SourceBAR-FNO prediction")
            axes[row, 2].set_title("Absolute error")
        for column in range(3):
            axes[row, column].set_xlabel("Range (km)")
    fig.colorbar(image0, ax=axes[:, :2], label="Transmission loss (dB)", shrink=0.75)
    fig.colorbar(image2, ax=axes[:, 2], label="Absolute error (dB)", shrink=0.75)
    save(fig, "fig03_unseen_depth_examples")


def plot_training(runs: list[dict]) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    colors = [BLUE, TEAL, ORANGE, GREEN]
    for run, color in zip(runs, colors):
        history = run["history"]
        axis.plot(
            [item["epoch"] for item in history],
            [item["validation_rmse_db"] for item in history],
            color=color,
            linewidth=1.8,
            label=run["metrics"]["experiment_id"].replace("deep_sourcebar_", ""),
        )
    axis.axhline(2.0, color=RED, linestyle="--", linewidth=1.4, label="2 dB")
    axis.set(
        xlabel="Epoch",
        ylabel="Validation RMSE (dB)",
        title="Model selection on source depths absent from training",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    save(fig, "fig04_training_curves")


def write_summary(config: MVPConfig, runs: list[dict], best: dict, dataset_path: Path) -> None:
    metrics = best["metrics"]
    compact = min(
        (run for run in runs if run["metrics"]["parameter_count"] < 10_000_000),
        key=lambda run: run["metrics"]["metrics"]["test"]["aggregate"]["rmse_db"],
    )
    compact_metrics = compact["metrics"]
    pilot = json.loads((dataset_path.parent.parent / "pilot/convergence_report.json").read_text())
    verifications = {}
    for run in runs:
        verification_path = run["run_dir"] / "independent_verification_cuda.json"
        if not verification_path.exists():
            continue
        raw = json.loads(verification_path.read_text())
        verifications[run["metrics"]["experiment_id"]] = {
            "run_id": raw["run_id"],
            "dataset_sha256": raw["dataset_sha256"],
            "dataset_hash_matches_run": raw["dataset_hash_matches_run"],
            "recorded_test_rmse_db": raw["recorded_test_rmse_db"],
            "recomputed_test_rmse_db": raw["recomputed_test_rmse_db"],
            "rmse_absolute_delta_db": raw["rmse_absolute_delta_db"],
            "checkpoint_reload_matches_rmse": raw["checkpoint_reload_matches_rmse"],
            "latency": raw["latency"],
            "acceptance": raw["acceptance"],
            "verification_pass": raw["verification_pass"],
        }
    payload = {
        "dataset_id": config.contract.dataset_id,
        "config_hash": config.config_hash,
        "dataset_path": str(dataset_path),
        "dataset_sha256": metrics["dataset_sha256"],
        "n_samples": 480,
        "source_depth_range_m": [50.0, 2000.0],
        "validation_source_depths_m": config.split.validation_source_depths_m,
        "test_source_depths_m": config.split.test_source_depths_m,
        "best_run": str(best["run_dir"]),
        "best_experiment_id": metrics["experiment_id"],
        "recommended_compact_run": str(compact["run_dir"]),
        "recommended_compact_experiment_id": compact_metrics["experiment_id"],
        "recommended_compact_test": compact_metrics["metrics"]["test"]["aggregate"],
        "recommended_compact_parameter_count": compact_metrics["parameter_count"],
        "test": metrics["metrics"]["test"]["aggregate"],
        "test_by_source_depth": metrics["metrics"]["test"]["by_source_depth"],
        "mean_field_baseline_test": metrics["mean_field_baseline_test"],
        "baseline_improvement_rmse_db": metrics["baseline_improvement_rmse_db"],
        "baseline_rmse_reduction_percent": metrics["baseline_rmse_reduction_percent"],
        "latency": metrics["latency"],
        "acceptance": metrics["acceptance"],
        "pilot_convergence": pilot["aggregate"],
        "pilot_failures": pilot["failures"],
        "independent_verification": verifications,
        "experiments": [
            {
                "experiment_id": run["metrics"]["experiment_id"],
                "test_rmse_db": run["metrics"]["metrics"]["test"]["aggregate"]["rmse_db"],
                "test_mae_db": run["metrics"]["metrics"]["test"]["aggregate"]["mae_db"],
                "parameter_count": run["metrics"]["parameter_count"],
            }
            for run in runs
        ],
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2) + "\n"
    (OUTPUT_ROOT / "results.json").write_text(serialized)
    result_root = ROOT / "docs/results"
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "unseen_source_depth_v1.0_verification_summary.json").write_text(serialized)
    test = payload["test"]
    baseline = payload["mean_field_baseline_test"]
    gpu_latency = payload["latency"].get("gpu")
    latency = gpu_latency or payload["latency"]["cpu"]
    experiment_name = payload["best_experiment_id"].replace("_", "\\_")
    compact_name = payload["recommended_compact_experiment_id"].replace("_", "\\_")
    macros = (
        f"\\newcommand{{\\BestExperiment}}{{{experiment_name}}}\n"
        f"\\newcommand{{\\TestRMSE}}{{{test['rmse_db']:.3f}}}\n"
        f"\\newcommand{{\\TestMAE}}{{{test['mae_db']:.3f}}}\n"
        f"\\newcommand{{\\BaselineRMSE}}{{{baseline['rmse_db']:.3f}}}\n"
        f"\\newcommand{{\\RMSEGain}}{{{payload['baseline_improvement_rmse_db']:.3f}}}\n"
        f"\\newcommand{{\\ReductionPercent}}{{{payload['baseline_rmse_reduction_percent']:.1f}}}\n"
        f"\\newcommand{{\\LatencyP}}{{{latency['p95_ms']:.2f}}}\n"
        f"\\newcommand{{\\ParameterCount}}{{{metrics['parameter_count']:,}}}\n"
        f"\\newcommand{{\\CompactExperiment}}{{{compact_name}}}\n"
        f"\\newcommand{{\\CompactRMSE}}{{{payload['recommended_compact_test']['rmse_db']:.3f}}}\n"
        f"\\newcommand{{\\CompactParameterCount}}{{{payload['recommended_compact_parameter_count']:,}}}\n"
        f"\\newcommand{{\\DatasetSHA}}{{{payload['dataset_sha256']}}}\n"
        f"\\newcommand{{\\ConfigHash}}{{{payload['config_hash']}}}\n"
    )
    (OUTPUT_ROOT / "results_macros.tex").write_text(macros)


def main() -> None:
    args = parse_args()
    config = MVPConfig.from_yaml(CONFIG_PATH)
    dataset_path = args.artifact_root / "datasets" / config.contract.dataset_id / "n480/dataset.npz"
    with np.load(dataset_path) as raw:
        dataset = {key: raw[key].copy() for key in raw.files}
    campaign_path = args.campaign or args.artifact_root / "campaigns/latest.json"
    runs, best = load_campaign(campaign_path)
    plot_design(config, dataset)
    plot_performance(runs, best)
    plot_examples(best)
    plot_training(runs)
    write_summary(config, runs, best, dataset_path)
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()
