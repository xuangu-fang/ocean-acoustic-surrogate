"""Generate the client-facing figures for the real-data-anchored technical report."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from ocean_acoustic_surrogate.config import MVPConfig

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "docs/technical_report/assets"
ARTIFACT_ROOT = Path(
    os.environ.get(
        "OCEAN_SURROGATE_ROOT",
        "/mnt/data/xuangu-fang/ocean-acoustics/projects/ocean-acoustic-surrogate",
    )
)
CONFIG_PATH = ROOT / "configs/realistic_terrain_mvp.yaml"
BLUE = "#2f6f9f"
NAVY = "#153f66"
TEAL = "#2a8c82"
GREEN = "#4d8f62"
ORANGE = "#d9822b"
RED = "#b04a4a"


def _save(fig: plt.Figure, name: str) -> None:
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSET_ROOT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(ASSET_ROOT / f"{name}.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def _load() -> tuple[MVPConfig, dict[str, np.ndarray], dict, dict, list[dict]]:
    config = MVPConfig.from_yaml(CONFIG_PATH)
    dataset_path = ARTIFACT_ROOT / "datasets" / config.contract.dataset_id / "n128/dataset.npz"
    with np.load(dataset_path) as raw:
        dataset = {key: raw[key].copy() for key in raw.files}
    pilot = json.loads(
        (
            ARTIFACT_ROOT
            / "datasets"
            / config.contract.dataset_id
            / "pilot/convergence_report.json"
        ).read_text()
    )
    campaign = json.loads((ARTIFACT_ROOT / "campaigns/latest.json").read_text())
    runs = []
    for record in campaign["experiments"]:
        run_dir = Path(record["run_dir"])
        metrics = json.loads((run_dir / "metrics.json").read_text())
        history = json.loads((run_dir / "history.json").read_text())
        runs.append({"record": record, "metrics": metrics, "history": history, "path": run_dir})
    return config, dataset, pilot, campaign, runs


def plot_environment(config: MVPConfig, data: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    speeds = data["ssp_speeds_mps"]
    ssp_depths = data["ssp_depths_m"]
    for profile in speeds[:: max(1, len(speeds) // 48)]:
        axes[0].plot(profile, ssp_depths, color=BLUE, alpha=0.13, linewidth=0.8)
    axes[0].plot(speeds.mean(0), ssp_depths, color=NAVY, linewidth=2.4, label="dataset mean")
    axes[0].invert_yaxis()
    axes[0].grid(alpha=0.18)
    axes[0].set(
        xlabel="Sound speed (m/s)",
        ylabel="Depth (m)",
        title="WOA23 June SSP and narrow perturbations",
    )
    axes[0].legend()

    for profile in config.contract.bathymetry.profiles:
        axes[1].plot(
            np.asarray(profile.ranges_m) / 1000.0,
            profile.depths_m,
            marker="o",
            markersize=3,
            label=profile.name.replace("gebco_", "").replace("_screened", "°"),
        )
    axes[1].invert_yaxis()
    axes[1].grid(alpha=0.18)
    axes[1].set(
        xlabel="Range (km)",
        ylabel="Bottom depth (m)",
        title="Four screened low-dimensional GEBCO profiles",
    )
    axes[1].legend(fontsize=8, ncol=2)
    _save(fig, "fig02_real_environment")


def plot_label_quality(pilot: dict) -> None:
    comparisons = pilot["aggregate"]
    labels = [f"{item['lower_rays']//1000:.1f}k→{item['upper_rays']//1000:.1f}k" for item in comparisons]
    mean = [item["mean_rmse_db"] for item in comparisons]
    worst = [item["worst_rmse_db"] for item in comparisons]
    rays = [int(value) for value in pilot["ray_counts"]]
    timing = [pilot["timing_seconds"][str(value)]["median"] for value in rays]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.5), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.36
    axes[0].bar(x - width / 2, mean, width, label="Mean RMSE", color=BLUE)
    axes[0].bar(x + width / 2, worst, width, label="Worst RMSE", color=ORANGE)
    axes[0].set_xticks(x, labels)
    axes[0].set(ylabel="Inter-level TL RMSE (dB)", title="Ray-count convergence")
    axes[0].grid(axis="y", alpha=0.18)
    axes[0].legend()
    axes[1].plot(np.asarray(rays) / 1000.0, timing, marker="o", color=TEAL, linewidth=2)
    axes[1].set(
        xlabel="Bellhop rays per field (thousand)",
        ylabel="Median wall time (s)",
        title="Reference-label cost",
    )
    axes[1].grid(alpha=0.18)
    _save(fig, "fig03_label_convergence")


def _run_label(experiment_id: str) -> str:
    labels = {
        "real_r1_small_terrain_fno": "Global FNO-S",
        "real_r2_terrain_fno": "Global FNO-L",
        "real_r3_terrain_anchor_fno": "Anchor FNO-S",
        "real_r4_terrain_anchor_large_fno": "Anchor FNO-L",
        "real_r5_anchor_data24": "Anchor FNO (24)",
        "real_r6_anchor_data48": "Anchor FNO (48)",
    }
    return labels.get(experiment_id, experiment_id)


def plot_main_results(runs: list[dict]) -> None:
    main = [run for run in runs if run["metrics"]["experiment_id"].startswith("real_r")][:4]
    baseline = main[0]["metrics"]["mean_field_baseline_test"]["rmse_db"]
    terrain = main[0]["metrics"]["terrain_mean_baseline_test"]["rmse_db"]
    labels = ["Global mean", "Terrain mean"] + [
        _run_label(run["metrics"]["experiment_id"]) for run in main
    ]
    values = [baseline, terrain] + [
        run["metrics"]["metrics"]["test"]["aggregate"]["rmse_db"] for run in main
    ]
    colors = ["#9aa5ad", "#6c8799"] + [BLUE, NAVY, TEAL, GREEN]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    bars = axes[0].bar(labels, values, color=colors)
    axes[0].axhline(2.0, color=RED, linestyle="--", label="2 dB gate")
    axes[0].set(ylabel="Held-out test RMSE (dB)", title="Baselines and operator variants")
    axes[0].tick_params(axis="x", rotation=24)
    axes[0].bar_label(bars, fmt="%.2f", fontsize=8)
    axes[0].legend()
    reductions = [run["metrics"]["baseline_rmse_reduction_percent"] for run in main]
    bars = axes[1].bar(
        [_run_label(run["metrics"]["experiment_id"]) for run in main],
        reductions,
        color=[BLUE, NAVY, TEAL, GREEN],
    )
    axes[1].set(ylabel="RMSE reduction vs global mean (%)", title="Relative improvement")
    axes[1].tick_params(axis="x", rotation=24)
    axes[1].bar_label(bars, fmt="%.1f%%", fontsize=8)
    axes[1].set_ylim(0, 100)
    _save(fig, "fig05_main_results")


def plot_data_ablation(runs: list[dict]) -> None:
    wanted = {
        "real_r5_anchor_data24": 24,
        "real_r6_anchor_data48": 48,
        "real_r3_terrain_anchor_fno": 96,
    }
    points = []
    for run in runs:
        experiment = run["metrics"]["experiment_id"]
        if experiment in wanted:
            points.append(
                (
                    wanted[experiment],
                    run["metrics"]["metrics"]["test"]["aggregate"]["rmse_db"],
                    run["metrics"]["metrics"]["test"]["aggregate"]["p90_sample_rmse_db"],
                )
            )
    points.sort()
    fig, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    x = [item[0] for item in points]
    y = [item[1] for item in points]
    p90 = [item[2] for item in points]
    axis.plot(x, y, marker="o", linewidth=2.2, color=BLUE, label="Aggregate RMSE")
    axis.plot(x, p90, marker="s", linewidth=1.8, color=ORANGE, label="P90 sample RMSE")
    axis.axhline(2.0, color=RED, linestyle="--", label="2 dB gate")
    axis.set(
        xlabel="Effective training fields",
        ylabel="Held-out test RMSE (dB)",
        title="Grouped data-size ablation on the same sealed test set",
        xticks=x,
    )
    axis.grid(alpha=0.18)
    axis.legend()
    _save(fig, "fig06_data_size_ablation")


def plot_training(runs: list[dict]) -> None:
    main = [run for run in runs if run["metrics"]["experiment_id"] in {
        "real_r1_small_terrain_fno",
        "real_r2_terrain_fno",
        "real_r3_terrain_anchor_fno",
        "real_r4_terrain_anchor_large_fno",
    }]
    fig, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for run in main:
        history = run["history"]
        axis.plot(
            [item["epoch"] for item in history],
            [item["validation_rmse_db"] for item in history],
            label=_run_label(run["metrics"]["experiment_id"]),
        )
    axis.axhline(2.0, color=RED, linestyle="--", label="2 dB gate")
    axis.set(xlabel="Epoch", ylabel="Validation RMSE (dB)", title="Validation trajectories")
    axis.grid(alpha=0.18)
    axis.legend(ncol=2, fontsize=8)
    _save(fig, "fig07_training_curves")


def plot_prediction_analysis(campaign: dict) -> None:
    run_dir = Path(campaign["best"]["run_dir"])
    with np.load(run_dir / "predictions.npz") as raw:
        prediction = raw["prediction_tl_db"]
        reference = raw["reference_tl_db"]
        valid = raw["valid_mask"].astype(bool)
        splits = raw["splits"].astype(str)
        sample_ids = raw["sample_ids"].astype(str)
        ranges = raw["ranges_m"] / 1000.0
        depths = raw["depths_m"]
        ssp = raw["ssp_speeds_mps"]
        ssp_depths = raw["ssp_depths_m"]
    test = np.flatnonzero(splits == "test")
    sample_rmse = np.asarray(
        [np.sqrt(np.mean((prediction[i][valid[i]] - reference[i][valid[i]]) ** 2)) for i in test]
    )
    order = np.argsort(sample_rmse)
    chosen = [test[order[len(order) // 2]], test[order[-1]]]
    fig, axes = plt.subplots(2, 5, figsize=(17.2, 7.4), constrained_layout=True)
    extent = [ranges[0], ranges[-1], depths[-1], depths[0]]
    tl_min, tl_max = np.percentile(reference[test][valid[test]], [1, 99])
    error_cap = max(2.0, float(np.percentile(np.abs(prediction[test] - reference[test])[valid[test]], 99)))
    for row, index in enumerate(chosen):
        axis = axes[row, 0]
        axis.plot(ssp[index], ssp_depths, color=BLUE)
        axis.set(xlabel="Sound speed (m/s)", ylabel="Depth (m)")
        axis.invert_yaxis()
        axis.set_title(f"{sample_ids[index]} environment")
        for column, (field, title, cmap, low, high) in enumerate(
            (
                (reference[index], "Bellhop reference", "viridis", tl_min, tl_max),
                (prediction[index], "Surrogate prediction", "viridis", tl_min, tl_max),
                (np.abs(prediction[index] - reference[index]), "Absolute error", "magma", 0, error_cap),
            ),
            start=1,
        ):
            image = axes[row, column].imshow(
                field, aspect="auto", extent=extent, cmap=cmap, vmin=low, vmax=high
            )
            axes[row, column].set(xlabel="Range (km)", ylabel="Depth (m)", title=title)
            fig.colorbar(image, ax=axes[row, column], label="dB")
        depth_index = int(np.argmin(np.abs(depths - 1000.0)))
        axes[row, 4].plot(ranges, reference[index, depth_index], label="Bellhop", color=NAVY)
        axes[row, 4].plot(ranges, prediction[index, depth_index], label="Surrogate", color=ORANGE)
        axes[row, 4].invert_yaxis()
        axes[row, 4].set(
            xlabel="Range (km)",
            ylabel="TL (dB)",
            title=f"1000 m slice | RMSE {sample_rmse[np.where(test == index)[0][0]]:.2f} dB",
        )
        axes[row, 4].legend(fontsize=8)
    _save(fig, "fig08_prediction_examples")

    errors = np.abs(prediction[test] - reference[test])
    masked = np.where(valid[test], errors, np.nan)
    mean_error = np.nanmean(masked, axis=0)
    p95_error = np.nanpercentile(masked, 95, axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.4), constrained_layout=True)
    for axis, field, title in zip(
        axes[:2], (mean_error, p95_error), ("Mean absolute error", "P95 absolute error")
    ):
        image = axis.imshow(field, aspect="auto", extent=extent, cmap="magma", vmin=0)
        axis.set(xlabel="Range (km)", ylabel="Depth (m)", title=title)
        fig.colorbar(image, ax=axis, label="dB")
    axes[2].hist(sample_rmse, bins=8, color=BLUE, alpha=0.85)
    axes[2].axvline(2.0, color=RED, linestyle="--", label="2 dB gate")
    axes[2].set(
        xlabel="Per-sample RMSE (dB)", ylabel="Test samples", title="Held-out sample distribution"
    )
    axes[2].legend()
    _save(fig, "fig09_error_analysis")


def plot_latency(runs: list[dict], pilot: dict) -> None:
    main = [run for run in runs if run["metrics"]["experiment_id"] in {
        "real_r1_small_terrain_fno",
        "real_r2_terrain_fno",
        "real_r3_terrain_anchor_fno",
        "real_r4_terrain_anchor_large_fno",
    }]
    labels = [_run_label(run["metrics"]["experiment_id"]) for run in main] + ["Bellhop 25.6k"]
    values = [run["metrics"]["latency"]["gpu"]["p95_ms"] for run in main] + [
        pilot["timing_seconds"]["25600"]["median"] * 1000.0
    ]
    fig, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    bars = axis.bar(labels, values, color=[BLUE, NAVY, TEAL, GREEN, "#9aa5ad"])
    axis.axhline(100.0, color=RED, linestyle="--", label="100 ms gate")
    axis.set_yscale("log")
    axis.set(ylabel="Batch=1 wall time (ms, log scale)", title="Surrogate latency vs reference solver")
    axis.tick_params(axis="x", rotation=20)
    axis.bar_label(bars, fmt="%.1f", fontsize=8)
    axis.legend()
    _save(fig, "fig10_latency")


def main() -> None:
    config, data, pilot, campaign, runs = _load()
    plot_environment(config, data)
    plot_label_quality(pilot)
    plot_main_results(runs)
    plot_data_ablation(runs)
    plot_training(runs)
    plot_prediction_analysis(campaign)
    plot_latency(runs, pilot)


if __name__ == "__main__":
    main()
