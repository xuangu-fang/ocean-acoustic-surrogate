"""Fast transfer validation on frozen real-environment Bellhop fields."""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from .features import TargetTransform
from .metrics import split_metrics, tl_metrics
from .models import build_model
from .training import _benchmark_latency, _predict_batches

matplotlib.use("Agg")
from matplotlib import pyplot as plt


@dataclass(frozen=True)
class FieldValidationData:
    features: np.ndarray
    targets_db: np.ndarray
    traditional_db: np.ndarray
    masks: np.ndarray
    splits: np.ndarray
    sample_ids: np.ndarray
    months: np.ndarray
    azimuths_deg: np.ndarray
    frequencies_hz: np.ndarray
    source_depths_m: np.ndarray
    ranges_m: np.ndarray
    depths_m: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _reference_features(
    metadata: dict,
    ranges_m: np.ndarray,
    depths_m: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    task = metadata["reference_task"]
    environment = task["environment"]
    bathymetry = environment["bathymetry"]
    if bathymetry is None:
        bottom_depths = np.full_like(ranges_m, float(environment["water_depth_m"]))
    else:
        points = bathymetry["points"]
        bottom_depths = np.interp(
            ranges_m,
            [point["range_m"] for point in points],
            [point["depth_m"] for point in points],
        ).astype(np.float32)
    ssp = environment["ssp"]
    if ssp.get("grid") is not None:
        raise ValueError("the quick adapter currently expects a range-independent SSP")
    ssp_points = ssp["points"]
    sound_speed = np.interp(
        depths_m,
        [point["depth_m"] for point in ssp_points],
        [point["speed_mps"] for point in ssp_points],
    ).astype(np.float32)
    shape = (len(depths_m), len(ranges_m))
    water_depth = max(float(environment["water_depth_m"]), 1.0)
    return np.stack(
        [
            np.broadcast_to(((sound_speed - 1500.0) / 50.0)[:, None], shape),
            np.broadcast_to((bottom_depths / water_depth)[None, :], shape),
            np.full(shape, float(task["frequency_hz"]) / 1000.0),
            np.full(shape, float(task["source_depth_m"]) / water_depth),
            valid_mask.astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32, copy=True)


def load_bashi_reuse_data(config: dict) -> tuple[FieldValidationData, dict]:
    """Load and crop one frozen Bashi slice without running Bellhop."""
    dataset_path = Path(config["source"]["dataset_path"])
    selection = config["slice"]
    with np.load(dataset_path) as raw:
        selected = np.ones(len(raw["sample_ids"]), dtype=bool)
        if selection.get("frequency_hz") is not None:
            selected &= np.isclose(raw["frequencies_hz"], float(selection["frequency_hz"]))
        if selection.get("source_depth_m") is not None:
            selected &= np.isclose(raw["source_depths_m"], float(selection["source_depth_m"]))
        if not np.any(selected):
            raise ValueError("the requested frequency/source-depth slice is absent")
        targets = raw["reference_tl_db"][selected].astype(np.float32)
        traditional = raw["traditional_tl_db"][selected].astype(np.float32)
        masks = raw["masks"][selected, 0].astype(bool)
        splits = raw["splits"][selected].astype(str)
        sample_ids = raw["sample_ids"][selected].astype(str)
        months = raw["months"][selected].astype(np.int64)
        azimuths = raw["azimuths_deg"][selected].astype(np.float64)
        frequencies = raw["frequencies_hz"][selected].astype(np.float64)
        source_depths = raw["source_depths_m"][selected].astype(np.float64)

    sample_root = dataset_path.parent / "samples"
    coordinate_path = sample_root / sample_ids[0] / "sample.npz"
    with np.load(coordinate_path) as sample:
        all_ranges = sample["ranges_m"].astype(np.float32)
        all_depths = sample["depths_m"].astype(np.float32)
    range_index = np.flatnonzero(all_ranges <= float(selection["range_end_m"]))
    depth_index = np.flatnonzero(all_depths <= float(selection["depth_end_m"]))
    if len(range_index) < 2 or len(depth_index) < 2:
        raise ValueError("the requested crop contains fewer than two grid cells")
    ranges = all_ranges[range_index]
    depths = all_depths[depth_index]
    targets = targets[:, depth_index][:, :, range_index]
    traditional = traditional[:, depth_index][:, :, range_index]
    masks = masks[:, depth_index][:, :, range_index] & np.isfinite(targets)

    features = []
    for sample_id, mask in zip(sample_ids, masks):
        metadata_path = sample_root / sample_id / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        features.append(_reference_features(metadata, ranges, depths, mask))
    features_np = np.stack(features)
    counts = {name: int(np.sum(splits == name)) for name in ("train", "validation", "test")}
    audit = {
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "selected_fields": int(selected.sum()),
        "split_counts": counts,
        "grid_shape": [len(depths), len(ranges)],
        "actual_range_m": [float(ranges[0]), float(ranges[-1])],
        "actual_depth_m": [float(depths[0]), float(depths[-1])],
        "frequencies_hz": sorted(np.unique(frequencies).tolist()),
        "source_depths_m": sorted(np.unique(source_depths).tolist()),
        "months": sorted(np.unique(months).tolist()),
        "azimuths_deg": sorted(np.unique(azimuths).tolist()),
        "feature_source": selection["feature_source"],
    }
    if min(counts.values()) == 0:
        raise ValueError(f"one or more frozen splits are empty: {counts}")
    return (
        FieldValidationData(
            features=features_np,
            targets_db=targets,
            traditional_db=traditional,
            masks=masks,
            splits=splits,
            sample_ids=sample_ids,
            months=months,
            azimuths_deg=azimuths,
            frequencies_hz=frequencies,
            source_depths_m=source_depths,
            ranges_m=ranges,
            depths_m=depths,
        ),
        audit,
    )


def _mean_field(targets: np.ndarray, masks: np.ndarray) -> np.ndarray:
    valid_count = masks.sum(axis=0)
    total = np.where(masks, targets, 0.0).sum(axis=0)
    global_mean = float(targets[masks].mean())
    return np.divide(
        total,
        valid_count,
        out=np.full_like(total, global_mean, dtype=np.float32),
        where=valid_count > 0,
    ).astype(np.float32)


def _azimuth_mean_prediction(data: FieldValidationData, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    prediction = np.empty_like(data.targets_db[test])
    test_azimuths = data.azimuths_deg[test]
    for azimuth in np.unique(data.azimuths_deg):
        train_group = train[np.isclose(data.azimuths_deg[train], azimuth)]
        test_group = np.flatnonzero(np.isclose(test_azimuths, azimuth))
        prediction[test_group] = _mean_field(
            data.targets_db[train_group], data.masks[train_group]
        )
    return prediction


def _geometry_mean_prediction(
    data: FieldValidationData, train: np.ndarray, test: np.ndarray
) -> np.ndarray:
    prediction = np.empty_like(data.targets_db[test])
    geometries = np.stack(
        (data.azimuths_deg, data.frequencies_hz, data.source_depths_m), axis=1
    )
    for geometry in np.unique(geometries, axis=0):
        all_group = np.all(np.isclose(geometries, geometry[None, :]), axis=1)
        train_group = train[all_group[train]]
        test_group = np.flatnonzero(all_group[test])
        prediction[test_group] = _mean_field(
            data.targets_db[train_group], data.masks[train_group]
        )
    return prediction


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((prediction - target) ** 2)[mask.expand_as(prediction)].mean()


def _fit_target_fields(
    data: FieldValidationData,
    train: np.ndarray,
    strategy: str,
) -> tuple[np.ndarray, float, dict]:
    if strategy == "global_train_mean":
        global_mean = _mean_field(data.targets_db[train], data.masks[train])
        mean_fields = np.broadcast_to(global_mean, data.targets_db.shape).copy()
        state = {"strategy": strategy, "global_mean_field_db": global_mean}
    elif strategy == "azimuth_conditioned_train_mean":
        mean_fields = np.empty_like(data.targets_db)
        anchors = {}
        for azimuth in np.unique(data.azimuths_deg):
            train_group = train[np.isclose(data.azimuths_deg[train], azimuth)]
            if len(train_group) == 0:
                raise ValueError(f"no training fields are available for azimuth {azimuth}")
            anchor = _mean_field(data.targets_db[train_group], data.masks[train_group])
            mean_fields[np.isclose(data.azimuths_deg, azimuth)] = anchor
            anchors[str(float(azimuth))] = anchor
        state = {"strategy": strategy, "mean_fields_by_azimuth_db": anchors}
    elif strategy == "geometry_conditioned_train_mean":
        mean_fields = np.empty_like(data.targets_db)
        anchors = {}
        geometries = np.stack(
            (data.azimuths_deg, data.frequencies_hz, data.source_depths_m), axis=1
        )
        for geometry in np.unique(geometries, axis=0):
            group = np.all(np.isclose(geometries, geometry[None, :]), axis=1)
            train_group = train[group[train]]
            if len(train_group) == 0:
                raise ValueError(f"no training fields are available for geometry {geometry}")
            anchor = _mean_field(data.targets_db[train_group], data.masks[train_group])
            mean_fields[group] = anchor
            key = f"az{geometry[0]:g}_f{geometry[1]:g}_zs{geometry[2]:g}"
            anchors[key] = anchor
        state = {"strategy": strategy, "mean_fields_by_geometry_db": anchors}
    else:
        raise ValueError(f"unknown target transform: {strategy}")
    residual = data.targets_db - mean_fields
    scale = max(float(np.std(residual[train][data.masks[train]])), 1.0)
    state["residual_scale_db"] = scale
    return mean_fields, scale, state


def _decode_prediction(
    normalized: torch.Tensor,
    mean_fields: np.ndarray,
    residual_scale_db: float,
) -> np.ndarray:
    residual = normalized[:, 0].numpy() * residual_scale_db
    return residual + mean_fields


def _plot_result(
    output: Path,
    data: FieldValidationData,
    test: np.ndarray,
    prediction: np.ndarray,
    history: list[dict],
) -> None:
    sample_metrics = split_metrics(
        data.targets_db[test], prediction[test], data.masks[test]
    )["per_sample"]
    order = np.argsort([item["rmse_db"] for item in sample_metrics])
    examples = [("median", int(order[len(order) // 2])), ("worst", int(order[-1]))]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    for row, (label, local_index) in enumerate(examples):
        index = test[local_index]
        reference = np.where(data.masks[index], data.targets_db[index], np.nan)
        estimate = np.where(data.masks[index], prediction[index], np.nan)
        error = np.where(data.masks[index], np.abs(estimate - reference), np.nan)
        title = f"{label.title()} test case | RMSE={sample_metrics[local_index]['rmse_db']:.2f} dB"
        for column, (field, name, cmap, limits) in enumerate(
            [
                (reference, "Bellhop reference TL", "viridis_r", (50, 110)),
                (estimate, "FNO prediction TL", "viridis_r", (50, 110)),
                (error, "absolute error", "magma", (0, 5)),
            ]
        ):
            image = axes[row, column].imshow(
                field,
                origin="upper",
                aspect="auto",
                extent=[
                    data.ranges_m[0] / 1000,
                    data.ranges_m[-1] / 1000,
                    data.depths_m[-1],
                    data.depths_m[0],
                ],
                cmap=cmap,
                vmin=limits[0],
                vmax=limits[1],
            )
            axes[row, column].set_title(f"{title}\n{name}" if column == 0 else name)
            axes[row, column].set_xlabel("Range (km)")
            axes[row, column].set_ylabel("Depth (m)")
            fig.colorbar(image, ax=axes[row, column], shrink=0.78, label="dB")
    axes[2, 0].plot([item["epoch"] for item in history], [item["train_loss"] for item in history])
    axes[2, 0].set_title("Training loss")
    axes[2, 1].plot(
        [item["epoch"] for item in history],
        [item["validation_rmse_db"] for item in history],
    )
    axes[2, 1].axhline(2.0, color="tab:red", linestyle="--", label="2 dB gate")
    axes[2, 1].set_title("Validation RMSE")
    axes[2, 1].legend()
    axes[2, 2].axis("off")
    for axis in axes[2, :2]:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run_bashi_reuse_validation(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text())
    data, audit = load_bashi_reuse_data(config)
    training = config["training"]
    seed = int(training["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indices = {
        name: np.flatnonzero(data.splits == name) for name in ("train", "validation", "test")
    }
    train = indices["train"]
    validation = indices["validation"]
    test = indices["test"]

    target_strategy = str(training.get("target_transform", "global_train_mean"))
    mean_fields, residual_scale_db, transform_state = _fit_target_fields(
        data, train, target_strategy
    )
    filled_targets = np.where(
        data.masks,
        data.targets_db,
        mean_fields,
    )
    normalized = ((filled_targets - mean_fields) / residual_scale_db).astype(np.float32)[:, None]
    features = torch.from_numpy(data.features)
    targets = torch.from_numpy(normalized)
    masks = torch.from_numpy(data.masks[:, None])
    loader = DataLoader(
        TensorDataset(features[train], targets[train], masks[train]),
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=device.type == "cuda",
    )
    model = build_model(config["model"], data.features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(training["max_epochs"]),
        eta_min=float(training["learning_rate"]) / 50,
    )
    history: list[dict] = []
    best_rmse = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, int(training["max_epochs"]) + 1):
        model.train()
        total_loss = 0.0
        for batch_features, batch_targets, batch_masks in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            batch_masks = batch_masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_features)
            loss = _masked_mse(prediction, batch_targets, batch_masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_features)
        scheduler.step()
        validation_normalized = _predict_batches(
            model, features[validation], device, int(training["batch_size"])
        )
        validation_db = _decode_prediction(
            validation_normalized,
            mean_fields[validation],
            residual_scale_db,
        )
        validation_metric = tl_metrics(
            data.targets_db[validation], validation_db, data.masks[validation]
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(train),
                "validation_rmse_db": validation_metric["rmse_db"],
                "validation_mae_db": validation_metric["mae_db"],
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )
        if validation_metric["rmse_db"] < best_rmse:
            best_rmse = validation_metric["rmse_db"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} train={history[-1]['train_loss']:.5f} "
                f"val_rmse={validation_metric['rmse_db']:.4f} dB",
                flush=True,
            )
        if stale_epochs >= int(training["early_stopping_patience"]):
            break
    training_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    prediction = _decode_prediction(
        _predict_batches(model, features, device, int(training["batch_size"])),
        mean_fields,
        residual_scale_db,
    )

    global_mean = _mean_field(data.targets_db[train], data.masks[train])
    mean_prediction = np.broadcast_to(global_mean, data.targets_db.shape)
    azimuth_prediction = _azimuth_mean_prediction(data, train, test)
    geometry_prediction = _geometry_mean_prediction(data, train, test)
    baselines = {
        "global_train_mean_field": split_metrics(
            data.targets_db[test], mean_prediction[test], data.masks[test]
        )["aggregate"],
        "azimuth_conditioned_train_mean": split_metrics(
            data.targets_db[test], azimuth_prediction, data.masks[test]
        )["aggregate"],
        "geometry_conditioned_train_mean": split_metrics(
            data.targets_db[test], geometry_prediction, data.masks[test]
        )["aggregate"],
        "existing_400_ray_bellhop": split_metrics(
            data.targets_db[test], data.traditional_db[test], data.masks[test]
        )["aggregate"],
    }
    metrics = {
        name: split_metrics(data.targets_db[index], prediction[index], data.masks[index])
        for name, index in indices.items()
    }
    focus = config.get("evaluation_focus")
    focus_test = test
    if focus:
        focus_mask = np.ones(len(test), dtype=bool)
        if focus.get("frequency_hz") is not None:
            focus_mask &= np.isclose(
                data.frequencies_hz[test], float(focus["frequency_hz"])
            )
        if focus.get("source_depth_m") is not None:
            focus_mask &= np.isclose(
                data.source_depths_m[test], float(focus["source_depth_m"])
            )
        focus_test = test[focus_mask]
        if len(focus_test) == 0:
            raise ValueError("evaluation_focus selects no frozen test fields")
        metrics["focus_test"] = split_metrics(
            data.targets_db[focus_test], prediction[focus_test], data.masks[focus_test]
        )
    per_azimuth = {}
    for azimuth in np.unique(data.azimuths_deg[test]):
        local = test[np.isclose(data.azimuths_deg[test], azimuth)]
        per_azimuth[str(float(azimuth))] = tl_metrics(
            data.targets_db[local], prediction[local], data.masks[local]
        )
    timing_transform = TargetTransform(
        mean_field_db=mean_fields[test[0]], residual_scale_db=residual_scale_db
    )
    gpu_latency = (
        _benchmark_latency(
            model,
            data.features[test[:1]],
            timing_transform,
            torch.device("cuda"),
            repeats=200,
        )
        if torch.cuda.is_available()
        else None
    )
    cpu_model = copy.deepcopy(model).cpu()
    cpu_latency = _benchmark_latency(
        cpu_model,
        data.features[test[:1]],
        timing_transform,
        torch.device("cpu"),
        repeats=40,
    )
    model.to(device)
    measured_latency = gpu_latency or cpu_latency
    test_metric = metrics["focus_test" if focus else "test"]["aggregate"]
    acceptance = config["acceptance"]
    gates = {
        "rmse_pass": test_metric["rmse_db"] <= float(acceptance["maximum_test_rmse_db"]),
        "mae_pass": test_metric["mae_db"] <= float(acceptance["maximum_test_mae_db"]),
        "latency_pass": measured_latency["p95_ms"]
        <= float(acceptance["maximum_p95_latency_ms"]),
    }
    gates["proxy_overall_pass"] = all(gates.values())

    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{config['experiment_id']}"
    run_dir = Path(config["storage"]["output_root"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "run_id": run_id,
        "scope": "quick_proxy_reusing_existing_bellhop_labels",
        "formal_acceptance": False,
        "formal_acceptance_limitations": [
            "existing labels are 500 Hz rather than the required 1 kHz",
            "existing reference uses 3,200 rays rather than the 25,600-ray MVP reference",
            "existing seabed is the ocean-field-project rock engineering assumption",
        ],
        "created_at": datetime.now(UTC).isoformat(),
        "config": config,
        "data_audit": audit,
        "source_repositories": {
            "ocean_acoustic_surrogate": _git_sha(Path(__file__).resolve().parents[2]),
            "ocean_field_project": _git_sha(Path(__file__).resolve().parents[3] / "ocean-field-project"),
        },
        "device": str(device),
        "hardware": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "training_seconds": training_seconds,
        "target_transform": {
            "strategy": target_strategy,
            "residual_scale_db": residual_scale_db,
        },
        "baselines": baselines,
        "metrics": metrics,
        "acceptance_metric_scope": "focus_test" if focus else "test",
        "test_by_azimuth": per_azimuth,
        "latency": {"gpu": gpu_latency, "cpu": cpu_latency},
        "proxy_acceptance": gates,
    }
    torch.save(
        {
            "model_state": best_state,
            "model_config": config["model"],
            "in_channels": data.features.shape[1],
            "target_transform": transform_state,
            "ranges_m": data.ranges_m,
            "depths_m": data.depths_m,
            "config": config,
        },
        run_dir / "model.pt",
    )
    np.savez_compressed(
        run_dir / "test_predictions.npz",
        prediction_tl_db=prediction[test],
        reference_tl_db=data.targets_db[test],
        valid_mask=data.masks[test],
        sample_ids=data.sample_ids[test],
        ranges_m=data.ranges_m,
        depths_m=data.depths_m,
    )
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    _plot_result(run_dir / "quick_validation.png", data, test, prediction, history)
    print(run_dir)
    return run_dir
