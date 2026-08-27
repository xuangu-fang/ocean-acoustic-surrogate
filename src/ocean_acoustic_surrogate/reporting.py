"""Lightweight committed plots and campaign summaries."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def plot_prediction_examples(run_dir: Path, output: Path) -> None:
    with np.load(run_dir / "predictions.npz") as raw:
        prediction = raw["prediction_tl_db"]
        reference = raw["reference_tl_db"]
        masks = raw["valid_mask"].astype(bool)
        splits = raw["splits"].astype(str)
        sample_ids = raw["sample_ids"].astype(str)
        ranges = raw["ranges_m"] / 1000.0
        depths = raw["depths_m"]
        ssp = raw["ssp_speeds_mps"]
        ssp_depths = raw["ssp_depths_m"]
    test = np.flatnonzero(splits == "test")
    rmse = np.asarray(
        [np.sqrt(np.mean((prediction[i][masks[i]] - reference[i][masks[i]]) ** 2)) for i in test]
    )
    order = np.argsort(rmse)
    chosen = [test[order[len(order) // 2]], test[order[-1]]]
    fig, axes = plt.subplots(2, 5, figsize=(18, 8), constrained_layout=True)
    extent = [ranges[0], ranges[-1], depths[-1], depths[0]]
    for row, index in enumerate(chosen):
        error = np.abs(prediction[index] - reference[index])
        axes[row, 0].plot(ssp[index], ssp_depths)
        axes[row, 0].invert_yaxis()
        axes[row, 0].set(xlabel="Sound speed (m/s)", ylabel="Depth (m)", title="SSP")
        for column, (field, title, cmap, low, high) in enumerate(
            (
                (reference[index], "Bellhop reference", "viridis", 45, 100),
                (prediction[index], "Surrogate", "viridis", 45, 100),
                (error, "Absolute error", "magma", 0, 5),
            ),
            start=1,
        ):
            image = axes[row, column].imshow(
                field,
                aspect="auto",
                extent=extent,
                cmap=cmap,
                vmin=low,
                vmax=high,
            )
            axes[row, column].set(xlabel="Range (km)", ylabel="Depth (m)", title=title)
            fig.colorbar(image, ax=axes[row, column], label="dB")
        depth_index = int(np.argmin(np.abs(depths - 1000.0)))
        axes[row, 4].plot(ranges, reference[index, depth_index], label="Bellhop")
        axes[row, 4].plot(ranges, prediction[index, depth_index], label="Prediction")
        axes[row, 4].set(
            xlabel="Range (km)",
            ylabel="TL (dB)",
            title=f"1000 m | RMSE={rmse[np.where(test == index)[0][0]]:.2f} dB",
        )
        axes[row, 4].invert_yaxis()
        axes[row, 4].legend(fontsize=8)
        axes[row, 0].text(
            0.03,
            0.96,
            sample_ids[index],
            transform=axes[row, 0].transAxes,
            va="top",
            fontsize=8,
        )
    fig.suptitle("Held-out incoherent Bellhop TL: median and worst test samples")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def write_campaign_summary(run_dirs: list[Path], output_json: Path) -> dict:
    records = []
    for run_dir in run_dirs:
        metrics = json.loads((run_dir / "metrics.json").read_text())
        history = json.loads((run_dir / "history.json").read_text())
        records.append(
            {
                "experiment_id": metrics["experiment_id"],
                "run_id": metrics["run_id"],
                "hypothesis": metrics["hypothesis"],
                "parameter_count": metrics["parameter_count"],
                "best_epoch": metrics["best_epoch"],
                "training_seconds": metrics["training_seconds"],
                "initial_train_loss": history[0]["train_loss"],
                "final_train_loss": history[-1]["train_loss"],
                "best_validation_rmse_db": min(
                    record["validation_rmse_db"] for record in history
                ),
                "test_rmse_db": metrics["metrics"]["test"]["aggregate"]["rmse_db"],
                "test_mae_db": metrics["metrics"]["test"]["aggregate"]["mae_db"],
                "test_p95_absolute_error_db": metrics["metrics"]["test"]["aggregate"][
                    "p95_absolute_error_db"
                ],
                "test_high_gradient_rmse_db": metrics["metrics"]["test"]["high_gradient"][
                    "rmse_db"
                ],
                "gpu_p95_latency_ms": metrics["latency"]["gpu"]["p95_ms"]
                if metrics["latency"]["gpu"]
                else None,
                "cpu_p95_latency_ms": metrics["latency"]["cpu"]["p95_ms"],
                "overall_pass": metrics["acceptance"]["overall_pass"],
                "run_dir": str(run_dir),
            }
        )
    best = min(records, key=lambda item: item["test_rmse_db"])
    summary = {"experiments": records, "best": best}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def plot_campaign(summary: dict, output: Path) -> None:
    records = summary["experiments"]
    labels = [record["experiment_id"] for record in records]
    rmse = [record["test_rmse_db"] for record in records]
    latency = [record["gpu_p95_latency_ms"] for record in records]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    bars = axes[0].bar(
        labels, rmse, color=["#2f6f9f" if value <= 2 else "#b04a4a" for value in rmse]
    )
    axes[0].axhline(2.0, color="black", linestyle="--", label="2 dB gate")
    axes[0].set(ylabel="Test RMSE (dB)", title="Accuracy iterations")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend()
    axes[0].bar_label(bars, fmt="%.2f")
    bars = axes[1].bar(labels, latency, color="#4d8f62")
    axes[1].axhline(100.0, color="black", linestyle="--", label="100 ms gate")
    axes[1].set(ylabel="GPU P95 end-to-end latency (ms)", title="Latency iterations")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend()
    axes[1].bar_label(bars, fmt="%.1f")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_training_curves(run_dirs: list[Path], output: Path) -> None:
    """Plot the optimization trace for every campaign round."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for run_dir in run_dirs:
        metrics = json.loads((run_dir / "metrics.json").read_text())
        history = json.loads((run_dir / "history.json").read_text())
        epochs = [record["epoch"] for record in history]
        label = metrics["experiment_id"]
        axes[0].plot(epochs, [record["train_loss"] for record in history], label=label)
        axes[1].plot(
            epochs,
            [record["validation_rmse_db"] for record in history],
            label=label,
        )
    axes[0].set(
        xlabel="Epoch",
        ylabel="Training objective",
        title="Training loss by iteration",
        yscale="log",
    )
    axes[1].set(
        xlabel="Epoch",
        ylabel="Validation RMSE (dB)",
        title="Validation trajectory",
    )
    axes[1].axhline(2.0, color="black", linestyle="--", linewidth=1, label="2 dB gate")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def commit_lightweight_results(summary: dict) -> None:
    root = project_root()
    output_json = root / "docs/results/campaign_summary.json"
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    plot_campaign(summary, root / "docs/assets/campaign_comparison.png")
    run_dirs = [Path(record["run_dir"]) for record in summary["experiments"]]
    plot_training_curves(run_dirs, root / "docs/assets/training_curves.png")
    best_run = Path(summary["best"]["run_dir"])
    plot_prediction_examples(best_run, root / "docs/assets/best_prediction_examples.png")
    shutil.copy2(best_run / "metrics.json", root / "docs/results/best_metrics.json")
