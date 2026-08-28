"""Generate the client-facing figures for the real-data-anchored technical report."""

from __future__ import annotations

import json
import os
from itertools import pairwise
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.io import netcdf_file

from ocean_acoustic_surrogate.config import MVPConfig

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "docs/technical_report/assets"
ARTIFACT_ROOT = Path(
    os.environ.get(
        "OCEAN_SURROGATE_ROOT",
        "/mnt/data/xuangu-fang/ocean-acoustics/projects/ocean-acoustic-surrogate",
    )
)
CONFIG_PATH = ROOT / "configs/realistic_seasonal_terrain_mvp.yaml"
GEBCO_PATH = Path(
    "/mnt/data/xuangu-fang/ocean-acoustics/shared/raw/gebco/GEBCO_2026/"
    "bashi_candidate_v0.1/gebco_2026_n21.578_s19.422_w120.349_e122.651.nc"
)
PILOT_DATASET_ID = "bashi_gebco_four_terrain_woa23_m03_m06_m12_v0.7_n384"
BLUE = "#2f6f9f"
NAVY = "#153f66"
TEAL = "#2a8c82"
GREEN = "#4d8f62"
ORANGE = "#d9822b"
RED = "#b04a4a"
MONTH_LABELS = {
    "woa23_march": "March",
    "woa23_june": "June",
    "woa23_december": "December",
}


def _save(fig: plt.Figure, name: str) -> None:
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSET_ROOT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(ASSET_ROOT / f"{name}.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def _load() -> tuple[MVPConfig, dict[str, np.ndarray], dict, dict, list[dict]]:
    config = MVPConfig.from_yaml(CONFIG_PATH)
    dataset_path = ARTIFACT_ROOT / "datasets" / config.contract.dataset_id / "n384/dataset.npz"
    with np.load(dataset_path) as raw:
        dataset = {key: raw[key].copy() for key in raw.files}
    pilot = json.loads(
        (
            ARTIFACT_ROOT
            / "datasets"
            / PILOT_DATASET_ID
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


def _box(axis: plt.Axes, x: float, y: float, width: float, height: float, title: str, body: str,
         color: str) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor=color,
            edgecolor=NAVY,
            linewidth=1.2,
        )
    )
    axis.text(x + width / 2, y + height * 0.67, title, ha="center", va="center",
              fontsize=11, fontweight="bold", color=NAVY)
    axis.text(x + width / 2, y + height * 0.33, body, ha="center", va="center", fontsize=8.5)


def _arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, color=BLUE, linewidth=1.5)
    )


def plot_pipeline() -> None:
    fig, axis = plt.subplots(figsize=(14.0, 3.2), constrained_layout=True)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    boxes = [
        (0.02, "Public data", "GEBCO 2026\nWOA23 + TEOS-10", "#e8f1f8"),
        (0.22, "Controlled domain", "4 terrain × 3 months\n384 balanced fields", "#edf6f4"),
        (0.42, "Reference labels", "Bellhop incoherent TL\n25,600 rays / field", "#fff3e6"),
        (0.62, "SeaBAR-FNO", "dual environment\n+ anisotropic residual operator", "#eaf3eb"),
        (0.82, "Sealed evaluation", "RMSE / MAE\nP95 latency", "#f1edf8"),
    ]
    for x, title, body, color in boxes:
        _box(axis, x, 0.25, 0.16, 0.5, title, body, color)
    for left, right in pairwise(boxes):
        _arrow(axis, (left[0] + 0.16, 0.5), (right[0], 0.5))
    _save(fig, "fig01_project_pipeline")


def plot_method() -> None:
    fig, axis = plt.subplots(figsize=(17.2, 6.4), constrained_layout=True)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    top = [
        (0.015, 0.13, "Raw environment", "SSP [B, 41]\nterrain [B, 256]", "#e8f1f8"),
        (0.175, 0.13, "Feature assembly", "normalize + replicate\n[B, 2, 96, 256]", "#edf6f4"),
        (0.335, 0.13, "Coordinates + lift", "append z,r; 1×1 conv\n[B, 32, 96, 256]", "#fff3e6"),
        (0.495, 0.13, "Replicate padding", "depth +8, range +16\n[B, 32, 104, 272]", "#f1edf8"),
        (0.655, 0.13, "4 SeaBAR blocks", "spectral + local + skip\n[B, 32, 104, 272]", "#eaf3eb"),
        (0.815, 0.17, "Crop + project", "32→128→1\n[B, 1, 96, 256]", "#e8f1f8"),
    ]
    for x, width, title, body, color in top:
        _box(axis, x, 0.66, width, 0.25, title, body, color)
    for left, right in pairwise(top):
        _arrow(axis, (left[0] + left[1], 0.785), (right[0], 0.785))

    _box(axis, 0.015, 0.20, 0.105, 0.22, "Block input", "$V_\\ell$\n[B,32,104,272]", "#f1edf8")
    _box(axis, 0.165, 0.29, 0.13, 0.20, "rFFT2", "[B,32,104,137]\nselect ±16 × 48", "#e8f1f8")
    _box(axis, 0.34, 0.29, 0.14, 0.20, "Complex mixing", "$R_\\ell(k_z,k_r)$\n32 input → 32 output", "#fff3e6")
    _box(axis, 0.525, 0.29, 0.12, 0.20, "irFFT2", "global update\n[B,32,104,272]", "#e8f1f8")
    _box(axis, 0.34, 0.03, 0.18, 0.20, "Local path", "1×1 convolution\n[B,32,104,272]", "#edf6f4")
    _box(axis, 0.70, 0.20, 0.14, 0.22, "Block update", "sum + GroupNorm\n+ skip + GELU", "#eaf3eb")
    _box(axis, 0.88, 0.20, 0.10, 0.22, "TL decode", "$\\bar y+s f$\n[B,96,256]", "#f1edf8")
    _arrow(axis, (0.12, 0.35), (0.165, 0.39))
    _arrow(axis, (0.295, 0.39), (0.34, 0.39))
    _arrow(axis, (0.48, 0.39), (0.525, 0.39))
    _arrow(axis, (0.645, 0.39), (0.70, 0.35))
    _arrow(axis, (0.12, 0.26), (0.34, 0.13))
    _arrow(axis, (0.52, 0.13), (0.70, 0.27))
    _arrow(axis, (0.84, 0.31), (0.88, 0.31))
    axis.text(
        0.50,
        0.52,
        "One anisotropic SeaBAR block (H=32, Kz=16, Kr=48); global and local updates retain the padded shape",
        ha="center",
        va="center",
        fontsize=9,
        color=NAVY,
        fontweight="bold",
    )
    axis.text(
        0.93,
        0.13,
        "stored train-only mean: [96, 256] + one scale",
        ha="center",
        va="center",
        fontsize=8,
        color=NAVY,
    )
    _save(fig, "fig04_model_architecture")


def plot_environment(config: MVPConfig, data: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.4, 4.9), constrained_layout=True)
    with netcdf_file(GEBCO_PATH, "r", mmap=False) as raw:
        longitude = np.asarray(raw.variables["lon"].data).copy()
        latitude = np.asarray(raw.variables["lat"].data).copy()
        elevation = np.asarray(raw.variables["elevation"].data, dtype=np.float32).copy()
    ocean = np.ma.masked_where(elevation >= 0, elevation / 1000.0)
    image = axes[0].contourf(
        longitude,
        latitude,
        ocean,
        levels=np.linspace(-5.0, 0.0, 21),
        cmap="Blues_r",
        extend="min",
    )
    axes[0].contour(longitude, latitude, elevation, levels=[0], colors="#4d5548", linewidths=0.8)
    center_lon, center_lat = 121.5, 20.5
    axes[0].scatter(center_lon, center_lat, marker="*", s=130, color=ORANGE,
                    edgecolor="white", linewidth=0.8, zorder=5, label="modeling origin")
    for azimuth in (30, 90, 210, 225):
        angle = np.deg2rad(azimuth)
        dlat = 50.0 * np.cos(angle) / 111.0
        dlon = 50.0 * np.sin(angle) / (111.0 * np.cos(np.deg2rad(center_lat)))
        axes[0].annotate(
            "",
            xy=(center_lon + dlon, center_lat + dlat),
            xytext=(center_lon, center_lat),
            arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 1.2, "alpha": 0.85},
        )
    axes[0].text(120.42, 21.46, "Taiwan side", color=NAVY, fontsize=8)
    axes[0].text(122.06, 19.52, "Luzon side", color=NAVY, fontsize=8)
    axes[0].set(
        xlabel="Longitude (°E)", ylabel="Latitude (°N)",
        title="Bashi Channel location and four 50 km directions",
    )
    axes[0].grid(alpha=0.16)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.colorbar(image, ax=axes[0], label="GEBCO elevation (km)")

    speeds = data["ssp_speeds_mps"]
    ssp_depths = data["ssp_depths_m"]
    ssp_groups = data["ssp_profiles"].astype(str)
    month_colors = {
        "woa23_march": BLUE,
        "woa23_june": ORANGE,
        "woa23_december": TEAL,
    }
    for group in ("woa23_march", "woa23_june", "woa23_december"):
        indices = np.flatnonzero(ssp_groups == group)
        selected = indices[np.linspace(0, len(indices) - 1, 24, dtype=int)]
        for index in selected:
            axes[1].plot(
                speeds[index], ssp_depths, color=month_colors[group],
                alpha=0.12, linewidth=0.75,
            )
        axes[1].plot(
            speeds[indices].mean(0), ssp_depths, color=month_colors[group], linewidth=2.4,
            label=f"{MONTH_LABELS[group]} mean",
        )
    axes[1].invert_yaxis()
    axes[1].grid(alpha=0.18)
    axes[1].set(
        xlabel="Sound speed (m/s)",
        ylabel="Depth (m)",
        title="Three representative WOA23 monthly SSP families",
    )
    axes[1].legend()

    for profile in config.contract.bathymetry.profiles:
        dense_range = np.linspace(0.0, 50_000.0, 401)
        dense_depth = np.interp(dense_range, profile.ranges_m, profile.depths_m)
        axes[2].plot(
            dense_range / 1000.0,
            dense_depth,
            linewidth=2.0,
            label=profile.name.replace("gebco_", "").replace("_screened", "°"),
        )
    axes[2].axhspan(0, 1990, color="#dcecf6", alpha=0.55, label="receiver domain")
    axes[2].invert_yaxis()
    axes[2].grid(alpha=0.18)
    axes[2].set(
        xlabel="Range (km)",
        ylabel="Bottom depth (m)",
        title="Screened low-dimensional bottom profiles",
    )
    axes[2].legend(fontsize=8, ncol=2)
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
        "seabar_a0_ssp_only": "SSP only",
        "seabar_a1_isotropic_spectrum": "Isotropic spectrum",
        "seabar_a2_no_boundary_padding": "No boundary padding",
        "seabar_fno": "SeaBAR-FNO",
    }
    return labels.get(experiment_id, experiment_id)


def plot_main_results(runs: list[dict]) -> None:
    main = runs
    baseline = main[0]["metrics"]["mean_field_baseline_test"]["rmse_db"]
    labels = ["Global mean"] + [
        _run_label(run["metrics"]["experiment_id"]) for run in main
    ]
    values = [baseline] + [
        run["metrics"]["metrics"]["test"]["aggregate"]["rmse_db"] for run in main
    ]
    colors = ["#9aa5ad", "#b7a6c9", TEAL, ORANGE, NAVY]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    bars = axes[0].bar(labels, values, color=colors)
    axes[0].axhline(2.0, color=RED, linestyle="--", label="2 dB gate")
    axes[0].set(ylabel="Held-out test RMSE (dB)", title="Global baseline and operator variants")
    axes[0].tick_params(axis="x", rotation=24)
    axes[0].bar_label(bars, fmt="%.2f", fontsize=8)
    axes[0].legend()
    reductions = [run["metrics"]["baseline_rmse_reduction_percent"] for run in main]
    bars = axes[1].bar(
        [_run_label(run["metrics"]["experiment_id"]) for run in main],
        reductions,
        color=["#b7a6c9", TEAL, ORANGE, NAVY],
    )
    axes[1].set(ylabel="RMSE reduction vs global mean (%)", title="Relative improvement")
    axes[1].tick_params(axis="x", rotation=24)
    axes[1].bar_label(bars, fmt="%.1f%%", fontsize=8)
    axes[1].set_ylim(0, 100)
    _save(fig, "fig05_main_results")


def plot_stratified_results(campaign: dict) -> None:
    metrics = json.loads((Path(campaign["best"]["run_dir"]) / "metrics.json").read_text())
    month = metrics["metrics"]["test"]["by_ssp_profile"]
    terrain = metrics["metrics"]["test"]["by_terrain"]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), constrained_layout=True)
    month_order = ["woa23_march", "woa23_june", "woa23_december"]
    bars = axes[0].bar(
        ["March", "June", "December"],
        [month[key]["rmse_db"] for key in month_order],
        color=[BLUE, ORANGE, TEAL],
    )
    axes[0].bar_label(bars, fmt="%.3f")
    axes[0].set(ylabel="Held-out RMSE (dB)", title="Performance by WOA23 month")
    terrain_order = sorted(terrain)
    bars = axes[1].bar(
        [key.replace("gebco_az", "az").replace("_screened", "°") for key in terrain_order],
        [terrain[key]["rmse_db"] for key in terrain_order],
        color=[BLUE, ORANGE, GREEN, RED],
    )
    axes[1].bar_label(bars, fmt="%.3f")
    axes[1].set(ylabel="Held-out RMSE (dB)", title="Performance by GEBCO terrain")
    for axis in axes:
        axis.axhline(2.0, color=RED, linestyle="--", label="2 dB gate")
        axis.set_ylim(0, 2.1)
        axis.grid(axis="y", alpha=0.18)
        axis.legend()
    _save(fig, "fig06_stratified_results")


def plot_training(runs: list[dict]) -> None:
    main = runs
    fig, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for run in main:
        history = run["history"]
        axis.plot(
            [item["epoch"] for item in history],
            [item["validation_rmse_db"] for item in history],
            label=_run_label(run["metrics"]["experiment_id"]),
        )
    axis.axhline(2.0, color=RED, linestyle="--", label="2 dB gate")
    axis.set_yscale("log")
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
        ssp_groups = raw["ssp_profiles"].astype(str)
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
        axis.set_title(
            f"{sample_ids[index]} | {MONTH_LABELS.get(ssp_groups[index], ssp_groups[index])}"
        )
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
    main = [run for run in runs if run["metrics"]["experiment_id"] != "seabar_a0_ssp_only"]
    labels = [_run_label(run["metrics"]["experiment_id"]) for run in main] + ["Bellhop 25.6k"]
    values = [run["metrics"]["latency"]["gpu"]["p95_ms"] for run in main] + [
        pilot["timing_seconds"]["25600"]["median"] * 1000.0
    ]
    fig, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    bars = axis.bar(labels, values, color=[TEAL, ORANGE, NAVY, "#9aa5ad"])
    axis.axhline(100.0, color=RED, linestyle="--", label="100 ms gate")
    axis.set_yscale("log")
    axis.set(ylabel="Batch=1 wall time (ms, log scale)", title="Surrogate latency vs reference solver")
    axis.tick_params(axis="x", rotation=20)
    axis.bar_label(bars, fmt="%.1f", fontsize=8)
    axis.legend()
    _save(fig, "fig10_latency")


def main() -> None:
    config, data, pilot, campaign, runs = _load()
    plot_pipeline()
    plot_environment(config, data)
    plot_label_quality(pilot)
    plot_method()
    plot_main_results(runs)
    plot_stratified_results(campaign)
    plot_training(runs)
    plot_prediction_analysis(campaign)
    plot_latency(runs, pilot)


if __name__ == "__main__":
    main()
