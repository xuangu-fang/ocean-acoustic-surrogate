"""Generate reproducible, publication-quality figures for the client whitepaper."""

from __future__ import annotations

import json
import os
from itertools import pairwise
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.use("Agg", force=True)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "whitepaper" / "assets"
ARTIFACT_ROOT = Path(
    os.environ.get("OCEAN_SURROGATE_ROOT", "/home/ubuntu/ocean-acoustic-surrogate-artifacts")
)

NAVY = "#153F66"
BLUE = "#2F6F9F"
TEAL = "#2A8C82"
GREEN = "#4D8F62"
ORANGE = "#D9822B"
RED = "#B04A4A"
PURPLE = "#745AA3"
LIGHT_BLUE = "#B8D3E8"
LIGHT_GRAY = "#E8EDF2"
DARK = "#24313D"

PARAMETER_NAMES = ["全局偏移", "温跃层幅度", "声道轴移动", "深层梯度"]
PARAMETER_UNITS = ["m/s", "m/s", "m", "m/s"]
EXPERIMENT_LABELS = ["R1\n小型 FNO", "R2\n增加模态", "R3\nPadding+残差", "R4\nHankel+局部", "R5\n结构损失"]


def configure_style() -> None:
    cjk_font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if cjk_font.exists():
        font_manager.fontManager.addfont(cjk_font)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.edgecolor": "#8A97A3",
            "axes.labelcolor": DARK,
            "axes.titlecolor": NAVY,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "xtick.color": DARK,
            "ytick.color": DARK,
            "grid.color": "#D8E0E7",
            "grid.alpha": 0.6,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_inputs() -> dict:
    artifact_manifest = json.loads((ROOT / "docs/results/artifact_manifest.json").read_text())
    campaign = json.loads((ROOT / "docs/results/campaign_summary.json").read_text())
    pilot = json.loads((ROOT / "docs/results/pilot_convergence_summary.json").read_text())
    dataset_summary = json.loads((ROOT / "docs/results/dataset_summary.json").read_text())
    dataset_path = Path(artifact_manifest["dataset"]["path"])
    with np.load(dataset_path) as raw:
        dataset = {key: raw[key].copy() for key in raw.files}
    runs = []
    for experiment in campaign["experiments"]:
        run_dir = Path(experiment["run_dir"])
        runs.append(
            {
                "summary": experiment,
                "metrics": json.loads((run_dir / "metrics.json").read_text()),
                "history": json.loads((run_dir / "history.json").read_text()),
                "run_dir": run_dir,
            }
        )
    best_run = Path(campaign["best"]["run_dir"])
    with np.load(best_run / "predictions.npz") as raw:
        predictions = {key: raw[key].copy() for key in raw.files}
    return {
        "artifact_manifest": artifact_manifest,
        "campaign": campaign,
        "pilot": pilot,
        "dataset_summary": dataset_summary,
        "dataset": dataset,
        "runs": runs,
        "predictions": predictions,
    }


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    subtitle: str,
    color: str,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.4,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height * 0.64, title, ha="center", va="center", fontsize=11, color=color, fontweight="bold")
    ax.text(xy[0] + width / 2, xy[1] + height * 0.30, subtitle, ha="center", va="center", fontsize=8.5, color=DARK)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.5,
            color="#6D7C88",
        )
    )


def figure_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(15, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.02, "冻结环境族", "4 参数 SSP\nLatin hypercube", NAVY),
        (0.185, "高质量标签", "Bellhop 非相干 TL\n25,600 rays", BLUE),
        (0.35, "数据封存", "384 / 64 / 64\n哈希 + 有效掩膜", TEAL),
        (0.515, "代理训练", "FNO2d 残差学习\n5 轮冻结消融", PURPLE),
        (0.68, "密封测试", "RMSE / MAE\n最差样本 / 高梯度", ORANGE),
        (0.845, "交付验收", "新进程重载\n精度 + P95 延迟", GREEN),
    ]
    width, height, y = 0.135, 0.48, 0.28
    for x, title, subtitle, color in boxes:
        add_box(ax, (x, y), width, height, title, subtitle, color)
    for left, right in pairwise(boxes):
        arrow(ax, (left[0] + width + 0.006, 0.52), (right[0] - 0.006, 0.52))
    ax.text(0.5, 0.93, "从数值标签到可审计代理模型的完整闭环", ha="center", fontsize=16, color=NAVY, fontweight="bold")
    ax.text(0.5, 0.08, "原则：固定参数、冻结测试集、先证明窄域可达，再讨论扩展", ha="center", fontsize=10.5, color=DARK)
    save(fig, "fig01_project_pipeline")


def figure_dataset_design(data: dict) -> None:
    dataset = data["dataset"]
    profiles = dataset["ssp_speeds_mps"]
    ssp_depths = dataset["ssp_depths_m"]
    parameters = dataset["parameters"]
    splits = dataset["splits"].astype(str)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for profile in profiles[::6]:
        axes[0, 0].plot(profile, ssp_depths, color=LIGHT_BLUE, alpha=0.28, linewidth=0.8)
    axes[0, 0].plot(profiles.mean(axis=0), ssp_depths, color=NAVY, linewidth=2.4, label="样本均值")
    axes[0, 0].invert_yaxis()
    axes[0, 0].set(xlabel="声速 (m/s)", ylabel="深度 (m)", title="窄域 SSP 样本族")
    axes[0, 0].legend()
    for i, ax in enumerate(axes.flat[1:5]):
        ax.hist(parameters[:, i], bins=16, color=[BLUE, TEAL, ORANGE, PURPLE][i], edgecolor="white")
        ax.set(title=PARAMETER_NAMES[i], xlabel=PARAMETER_UNITS[i], ylabel="样本数")
        ax.grid(axis="y")
    counts = [np.sum(splits == name) for name in ("train", "validation", "test")]
    axes[1, 2].bar(["训练", "验证", "测试"], counts, color=[BLUE, ORANGE, GREEN])
    axes[1, 2].set(title="冻结数据划分", ylabel="样本数", ylim=(0, 430))
    for index, value in enumerate(counts):
        axes[1, 2].text(index, value + 9, str(value), ha="center", color=DARK, fontweight="bold")
    axes[1, 2].grid(axis="y")
    fig.suptitle("数据设计：低维、平滑、可解释、同分布", color=NAVY, fontsize=16, fontweight="bold")
    save(fig, "fig02_dataset_design")


def figure_label_quality(data: dict) -> None:
    pilot = data["pilot"]
    summary = data["dataset_summary"]
    comparisons = pilot["comparisons"]
    upper = np.asarray([item["upper_rays"] for item in comparisons])
    mean = np.asarray([item["mean_rmse_db"] for item in comparisons])
    worst = np.asarray([item["worst_rmse_db"] for item in comparisons])
    timing = summary["bellhop_wall_seconds"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), constrained_layout=True)
    axes[0].plot(upper, mean, marker="o", linewidth=2.2, color=BLUE, label="8 个 SSP 平均")
    axes[0].plot(upper, worst, marker="s", linewidth=2.2, color=ORANGE, label="最差逐场")
    axes[0].axhline(2.0, linestyle="--", color=RED, label="最终 2 dB 门槛")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(upper, [f"{value:,}" for value in upper])
    axes[0].set(xlabel="较高一级射线数", ylabel="相邻层级 TL RMSE (dB)", title="射线数收敛审计", ylim=(0, 2.15))
    axes[0].grid()
    axes[0].legend()
    for x, y in zip(upper, worst):
        axes[0].annotate(f"{y:.3f}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    values = [timing["median"], timing["p95"]]
    bars = axes[1].bar(["中位数", "P95"], values, color=[TEAL, GREEN], width=0.55)
    axes[1].set(title="25,600-ray 正式标签生成耗时", ylabel="单场 Bellhop 时间 (s)", ylim=(0, 11))
    axes[1].grid(axis="y")
    axes[1].bar_label(bars, fmt="%.2f s", padding=4, color=DARK)
    axes[1].text(0.5, 0.82, "512 / 512 成功", transform=axes[1].transAxes, ha="center", color=NAVY, fontsize=15, fontweight="bold")
    axes[1].text(0.5, 0.70, f"累计 {timing['total'] / 60:.2f} min", transform=axes[1].transAxes, ha="center", color=DARK, fontsize=11)
    fig.suptitle("标签可信度：误差收敛与生成稳定性", color=NAVY, fontsize=16, fontweight="bold")
    save(fig, "fig03_label_quality")


def figure_model_architecture() -> None:
    fig, ax = plt.subplots(figsize=(15, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_box(ax, (0.025, 0.56), 0.13, 0.25, "一维 SSP", "33 点平滑剖面", NAVY)
    add_box(ax, (0.19, 0.56), 0.15, 0.25, "特征构建", "插值至 96 深度\n沿距离复制 + 坐标", BLUE)
    add_box(ax, (0.39, 0.56), 0.18, 0.25, "Lift", "1×1 Conv\n1/2 通道 → hidden", TEAL)
    add_box(ax, (0.62, 0.56), 0.18, 0.25, "4× FNO Block", "FFT → 截断模态 → 复权重\n+ 局部分支 + Norm + GELU", PURPLE)
    add_box(ax, (0.845, 0.56), 0.13, 0.25, "二维残差", "96 × 256", ORANGE)
    for a, b in [((0.155, 0.685), (0.19, 0.685)), ((0.34, 0.685), (0.39, 0.685)), ((0.57, 0.685), (0.62, 0.685)), ((0.80, 0.685), (0.845, 0.685))]:
        arrow(ax, a, b)
    add_box(ax, (0.25, 0.13), 0.21, 0.23, "训练集平均 TL", "仅使用 train 拟合\n固定空间先验", GREEN)
    add_box(ax, (0.56, 0.13), 0.21, 0.23, "反归一化与相加", "TL = mean field\n+ σres × predicted residual", RED)
    arrow(ax, (0.46, 0.245), (0.56, 0.245))
    arrow(ax, (0.91, 0.56), (0.72, 0.36))
    ax.text(0.5, 0.94, "各向异性 FNO：学习 SSP 引起的声场残差", ha="center", fontsize=16, color=NAVY, fontweight="bold")
    ax.text(0.5, 0.035, "R3: hidden=32, modes=(16,48), 4 层, padding=(8,16), 6.30M 参数", ha="center", fontsize=10.5, color=DARK)
    save(fig, "fig04_model_architecture")


def figure_training(data: dict) -> None:
    runs = data["runs"]
    colors = [BLUE, TEAL, ORANGE, GREEN, PURPLE]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    for run, color, label in zip(runs, colors, EXPERIMENT_LABELS):
        history = run["history"]
        epoch = [item["epoch"] for item in history]
        axes[0].plot(epoch, [item["train_loss"] for item in history], color=color, label=label.replace("\n", " "))
        axes[1].plot(epoch, [item["validation_rmse_db"] for item in history], color=color, label=label.replace("\n", " "))
    axes[0].set_yscale("log")
    axes[0].set(xlabel="Epoch", ylabel="训练目标", title="优化轨迹（R5 含额外结构项）")
    axes[1].axhline(2, linestyle="--", color=RED, label="2 dB 门槛")
    axes[1].set(xlabel="Epoch", ylabel="验证 RMSE (dB)", title="验证集轨迹", ylim=(0.45, 2.05))
    for ax in axes:
        ax.grid()
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle("五轮训练过程：统一数据、统一预算、统一验证规则", color=NAVY, fontsize=16, fontweight="bold")
    save(fig, "fig05_training_curves")


def figure_ablation(data: dict) -> None:
    records = data["campaign"]["experiments"]
    metrics = [run["metrics"] for run in data["runs"]]
    x = np.arange(len(records))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    width = 0.24
    rmse = [record["test_rmse_db"] for record in records]
    mae = [record["test_mae_db"] for record in records]
    difficult = [record["test_high_gradient_rmse_db"] for record in records]
    axes[0, 0].bar(x - width, rmse, width, label="整体 RMSE", color=BLUE)
    axes[0, 0].bar(x, mae, width, label="整体 MAE", color=TEAL)
    axes[0, 0].bar(x + width, difficult, width, label="高梯度 RMSE", color=ORANGE)
    axes[0, 0].axhline(2, linestyle="--", color=RED)
    axes[0, 0].set(title="精度消融", ylabel="误差 (dB)", xticks=x, xticklabels=EXPERIMENT_LABELS, ylim=(0, 2.15))
    axes[0, 0].legend(ncol=3, fontsize=8)
    gpu = [record["gpu_p95_latency_ms"] for record in records]
    cpu = [record["cpu_p95_latency_ms"] for record in records]
    axes[0, 1].bar(x - 0.18, gpu, 0.36, label="A100 P95", color=GREEN)
    axes[0, 1].bar(x + 0.18, cpu, 0.36, label="CPU P95", color=PURPLE)
    axes[0, 1].axhline(100, linestyle="--", color=RED, label="100 ms 门槛")
    axes[0, 1].set(title="端到端延迟消融", ylabel="P95 (ms)", xticks=x, xticklabels=EXPERIMENT_LABELS, ylim=(0, 160))
    axes[0, 1].legend(fontsize=8)
    params = np.asarray([record["parameter_count"] for record in records]) / 1e6
    seconds = [record["training_seconds"] for record in records]
    axes[1, 0].bar(x - 0.18, params, 0.36, color=BLUE, label="参数量 (M)")
    ax2 = axes[1, 0].twinx()
    ax2.bar(x + 0.18, seconds, 0.36, color=ORANGE, alpha=0.8, label="训练时间 (s)")
    axes[1, 0].set(title="复杂度代价", ylabel="参数量 (M)", xticks=x, xticklabels=EXPERIMENT_LABELS, ylim=(0, 7))
    ax2.set_ylabel("训练时间 (s)", color=ORANGE)
    handles1, labels1 = axes[1, 0].get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    axes[1, 0].legend(handles1 + handles2, labels1 + labels2, fontsize=8)
    baseline = float(metrics[0]["mean_field_baseline_test"]["rmse_db"])
    improvement = (1 - np.asarray(rmse) / baseline) * 100
    bars = axes[1, 1].bar(x, improvement, color=[BLUE, TEAL, ORANGE, GREEN, PURPLE])
    axes[1, 1].set(title="相对训练均值场的 RMSE 降幅", ylabel="改善 (%)", xticks=x, xticklabels=EXPERIMENT_LABELS, ylim=(0, 70))
    axes[1, 1].bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    for ax in axes.flat:
        ax.grid(axis="y")
    fig.suptitle("消融结论：更多结构并不等于更优交付", color=NAVY, fontsize=16, fontweight="bold")
    save(fig, "fig06_ablation_study")


def error_inputs(data: dict) -> dict:
    dataset = data["dataset"]
    pred = data["predictions"]
    splits = dataset["splits"].astype(str)
    train = np.flatnonzero(splits == "train")
    test = np.flatnonzero(splits == "test")
    tl = dataset["tl_db"].astype(np.float32)
    mask = dataset["valid_mask"].astype(bool)
    count = mask[train].sum(axis=0)
    mean_field = np.divide(
        np.where(mask[train], tl[train], 0).sum(axis=0),
        count,
        out=np.zeros_like(tl[0]),
        where=count > 0,
    )
    prediction = pred["prediction_tl_db"].astype(np.float32)
    reference = pred["reference_tl_db"].astype(np.float32)
    error = np.abs(prediction[test] - reference[test])
    baseline_error = np.abs(mean_field[None] - reference[test])
    valid = mask[test]
    model_per_sample = np.asarray([np.sqrt(np.mean((prediction[i][mask[i]] - reference[i][mask[i]]) ** 2)) for i in test])
    baseline_per_sample = np.asarray([np.sqrt(np.mean((mean_field[mask[i]] - reference[i][mask[i]]) ** 2)) for i in test])
    return {
        "test": test,
        "reference": reference,
        "prediction": prediction,
        "mask": mask,
        "error": error,
        "baseline_error": baseline_error,
        "valid": valid,
        "model_per_sample": model_per_sample,
        "baseline_per_sample": baseline_per_sample,
        "ranges_km": dataset["ranges_m"] / 1000,
        "depths_m": dataset["depths_m"],
        "sample_ids": dataset["sample_ids"].astype(str),
        "profiles": dataset["ssp_speeds_mps"],
        "ssp_depths": dataset["ssp_depths_m"],
    }


def ecdf(values: np.ndarray, limit: int = 180_000) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(values)
    if len(values) > limit:
        index = np.linspace(0, len(values) - 1, limit).astype(int)
        values = values[index]
    probability = np.linspace(1 / len(values), 1, len(values))
    return values, probability


def figure_error_distribution(data: dict) -> None:
    values = error_inputs(data)
    model_error = values["error"][values["valid"]]
    baseline_error = values["baseline_error"][values["valid"]]
    x_model, y_model = ecdf(model_error)
    x_base, y_base = ecdf(baseline_error)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    axes[0].plot(x_base, y_base * 100, color="#8A97A3", linewidth=2, label="训练均值场 baseline")
    axes[0].plot(x_model, y_model * 100, color=BLUE, linewidth=2.4, label="R3 FNO")
    axes[0].axvline(2, color=RED, linestyle="--", label="2 dB")
    axes[0].set(xlabel="有效网格绝对误差 (dB)", ylabel="累计网格比例 (%)", title="绝对误差累积分布", xlim=(0, 5), ylim=(0, 100))
    axes[0].grid()
    axes[0].legend()
    order = np.argsort(values["baseline_per_sample"])
    axes[1].plot(np.arange(1, 65), values["baseline_per_sample"][order], color="#8A97A3", linewidth=2, label="Baseline")
    axes[1].plot(np.arange(1, 65), values["model_per_sample"][order], color=TEAL, linewidth=2.4, label="R3 FNO")
    axes[1].axhline(2, color=RED, linestyle="--", label="2 dB")
    axes[1].set(xlabel="测试样本（按 baseline 难度排序）", ylabel="逐样本 RMSE (dB)", title="64 条测试样本逐场误差", ylim=(0, 2.1))
    axes[1].grid()
    axes[1].legend()
    fig.suptitle("模型收益不是平均值幻觉：整体分布与逐样本均改善", color=NAVY, fontsize=16, fontweight="bold")
    save(fig, "fig07_error_distribution")


def percentile_map(error: np.ndarray, valid: np.ndarray, quantile: float) -> np.ndarray:
    flat_error = error.reshape(error.shape[0], -1)
    flat_valid = valid.reshape(valid.shape[0], -1)
    output = np.full(flat_error.shape[1], np.nan, dtype=np.float32)
    for column in range(flat_error.shape[1]):
        selected = flat_error[flat_valid[:, column], column]
        if len(selected):
            output[column] = np.percentile(selected, quantile)
    return output.reshape(error.shape[1:])


def figure_error_maps(data: dict) -> None:
    values = error_inputs(data)
    error = values["error"]
    valid = values["valid"]
    masked = np.where(valid, error, np.nan)
    count = valid.sum(axis=0)
    mean_map = np.divide(
        np.where(valid, error, 0).sum(axis=0),
        count,
        out=np.full(error.shape[1:], np.nan, dtype=np.float32),
        where=count > 0,
    )
    p95_map = percentile_map(error, valid, 95)
    ranges = values["ranges_km"]
    depths = values["depths_m"]
    extent = [ranges[0], ranges[-1], depths[-1], depths[0]]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    image = axes[0, 0].imshow(mean_map, aspect="auto", extent=extent, cmap="magma", vmin=0, vmax=1.2)
    axes[0, 0].set(title="测试集逐网格平均绝对误差", xlabel="距离 (km)", ylabel="深度 (m)")
    fig.colorbar(image, ax=axes[0, 0], label="dB")
    image = axes[0, 1].imshow(p95_map, aspect="auto", extent=extent, cmap="magma", vmin=0, vmax=5)
    axes[0, 1].set(title="测试集逐网格 P95 绝对误差", xlabel="距离 (km)", ylabel="深度 (m)")
    fig.colorbar(image, ax=axes[0, 1], label="dB")
    axes[1, 0].plot(ranges, np.nanmean(masked, axis=(0, 1)), color=BLUE, linewidth=2)
    axes[1, 0].fill_between(ranges, 0, np.nanpercentile(masked, 95, axis=(0, 1)), color=LIGHT_BLUE, alpha=0.55, label="P95")
    axes[1, 0].set(title="误差随距离变化", xlabel="距离 (km)", ylabel="绝对误差 (dB)")
    axes[1, 0].grid()
    axes[1, 0].legend()
    axes[1, 1].plot(np.nanmean(masked, axis=(0, 2)), depths, color=TEAL, linewidth=2)
    axes[1, 1].fill_betweenx(depths, 0, np.nanpercentile(masked, 95, axis=(0, 2)), color="#BFE1DC", alpha=0.65, label="P95")
    axes[1, 1].invert_yaxis()
    axes[1, 1].set(title="误差随深度变化", xlabel="绝对误差 (dB)", ylabel="深度 (m)")
    axes[1, 1].grid()
    axes[1, 1].legend()
    fig.suptitle("误差空间分解：主要集中在窄会聚线而非大面积偏差", color=NAVY, fontsize=16, fontweight="bold")
    save(fig, "fig08_error_spatial_analysis")


def figure_worst_case(data: dict) -> None:
    values = error_inputs(data)
    local = int(np.argmax(values["model_per_sample"]))
    index = int(values["test"][local])
    reference = values["reference"][index]
    prediction = values["prediction"][index]
    valid = values["mask"][index]
    error = np.where(valid, np.abs(prediction - reference), np.nan)
    worst_z, worst_r = np.unravel_index(np.nanargmax(error), error.shape)
    ranges = values["ranges_km"]
    depths = values["depths_m"]
    extent = [ranges[0], ranges[-1], depths[-1], depths[0]]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    image = axes[0, 0].imshow(reference, aspect="auto", extent=extent, cmap="viridis", vmin=45, vmax=100)
    axes[0, 0].set(title="Bellhop 参考", xlabel="距离 (km)", ylabel="深度 (m)")
    fig.colorbar(image, ax=axes[0, 0], label="TL (dB)")
    image = axes[0, 1].imshow(prediction, aspect="auto", extent=extent, cmap="viridis", vmin=45, vmax=100)
    axes[0, 1].set(title="R3 预测", xlabel="距离 (km)", ylabel="深度 (m)")
    fig.colorbar(image, ax=axes[0, 1], label="TL (dB)")
    image = axes[0, 2].imshow(error, aspect="auto", extent=extent, cmap="magma", vmin=0, vmax=5)
    axes[0, 2].scatter(ranges[worst_r], depths[worst_z], marker="x", color="cyan", s=55, linewidth=2)
    axes[0, 2].set(title="绝对误差与最差有效点", xlabel="距离 (km)", ylabel="深度 (m)")
    fig.colorbar(image, ax=axes[0, 2], label="dB")
    axes[1, 0].plot(ranges, reference[worst_z], color=NAVY, label="Bellhop")
    axes[1, 0].plot(ranges, prediction[worst_z], color=ORANGE, label="预测")
    axes[1, 0].invert_yaxis()
    axes[1, 0].set(title=f"最差点深度切片：{depths[worst_z]:.1f} m", xlabel="距离 (km)", ylabel="TL (dB)")
    axes[1, 0].grid()
    axes[1, 0].legend()
    axes[1, 1].plot(reference[:, worst_r], depths, color=NAVY, label="Bellhop")
    axes[1, 1].plot(prediction[:, worst_r], depths, color=ORANGE, label="预测")
    axes[1, 1].invert_yaxis()
    axes[1, 1].invert_xaxis()
    axes[1, 1].set(title=f"最差点距离切片：{ranges[worst_r]:.2f} km", xlabel="TL (dB)", ylabel="深度 (m)")
    axes[1, 1].grid()
    axes[1, 1].legend()
    finite_error = error[np.isfinite(error)]
    axes[1, 2].hist(finite_error, bins=np.linspace(0, 5, 45), color=PURPLE, edgecolor="white")
    axes[1, 2].axvline(2, color=RED, linestyle="--", label="2 dB")
    axes[1, 2].set(title="该样本有效网格误差分布（截断至 5 dB）", xlabel="绝对误差 (dB)", ylabel="网格数")
    axes[1, 2].set_xlim(0, 5)
    axes[1, 2].legend()
    fig.suptitle(
        f"最差测试样本 {values['sample_ids'][index]}：逐场 RMSE={values['model_per_sample'][local]:.3f} dB",
        color=NAVY,
        fontsize=16,
        fontweight="bold",
    )
    save(fig, "fig09_worst_case")


def figure_acceptance(data: dict) -> None:
    r3 = data["runs"][2]["metrics"]
    cpu_verify = json.loads(
        (
            ARTIFACT_ROOT
            / "runs/20260827T104036Z_r1_fno_small/independent_verification_cpu.json"
        ).read_text()
    )
    names = ["R3 CUDA\nRMSE", "R3 CUDA\nMAE", "R3 CUDA\nP95", "R1 CPU\nRMSE", "R1 CPU\nMAE", "R1 CPU\nP95"]
    raw = [
        r3["metrics"]["test"]["aggregate"]["rmse_db"],
        r3["metrics"]["test"]["aggregate"]["mae_db"],
        r3["latency"]["gpu"]["p95_ms"],
        cpu_verify["test_metrics"]["aggregate"]["rmse_db"],
        cpu_verify["test_metrics"]["aggregate"]["mae_db"],
        cpu_verify["latency"]["p95_ms"],
    ]
    gates = [2, 2, 100, 2, 2, 100]
    ratio = np.asarray(raw) / np.asarray(gates)
    colors = [BLUE, TEAL, GREEN, BLUE, TEAL, GREEN]
    fig, ax = plt.subplots(figsize=(13.5, 5.4))
    bars = ax.bar(np.arange(6), ratio * 100, color=colors, width=0.66)
    ax.axhline(100, color=RED, linestyle="--", linewidth=2, label="验收上限")
    ax.set(xticks=np.arange(6), xticklabels=names, ylabel="占验收上限比例 (%)", title="双模型独立验收余量", ylim=(0, 110))
    for bar, value, gate in zip(bars, raw, gates):
        unit = "ms" if gate == 100 else "dB"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{value:.3f} {unit}", ha="center", fontsize=9, color=DARK)
    ax.grid(axis="y")
    ax.legend()
    fig.text(0.5, 0.01, "R3 为 A100 精度优胜；R1 为无需 GPU 的便携 MVP。全部柱低于 100% 即通过。", ha="center", color=DARK, fontsize=10)
    save(fig, "fig10_acceptance_margin")


def main() -> None:
    configure_style()
    data = load_inputs()
    figure_pipeline()
    figure_dataset_design(data)
    figure_label_quality(data)
    figure_model_architecture()
    figure_training(data)
    figure_ablation(data)
    figure_error_distribution(data)
    figure_error_maps(data)
    figure_worst_case(data)
    figure_acceptance(data)
    print(f"generated figures in {OUT}")


if __name__ == "__main__":
    main()
