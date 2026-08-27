"""Training, evaluation, and latency benchmarking for one frozen experiment."""

from __future__ import annotations

import copy
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import MVPConfig
from .features import TargetTransform, build_features, interpolate_ssp
from .metrics import high_gradient_mask, split_metrics, tl_metrics
from .models import build_model


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = mask.expand_as(values)
    return values[selected].mean()


def _loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    config: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    squared = (prediction - target) ** 2
    value_loss = _masked_mean(squared, mask)
    convergence_weight = float(config.get("convergence_weight", 0.0))
    convergence_loss = prediction.new_zeros(())
    if convergence_weight > 0:
        dz = torch.abs(torch.diff(target, dim=-2, prepend=target[..., :1, :]))
        dr = torch.abs(torch.diff(target, dim=-1, prepend=target[..., :, :1]))
        magnitude = dz + dr
        threshold = torch.quantile(
            magnitude[mask.expand_as(magnitude)], float(config.get("convergence_quantile", 0.9))
        )
        difficult = mask.expand_as(magnitude) & (magnitude >= threshold)
        if torch.any(difficult):
            convergence_loss = squared[difficult].mean()

    gradient_weight = float(config.get("gradient_weight", 0.0))
    gradient_loss = prediction.new_zeros(())
    if gradient_weight > 0:
        pred_dz = torch.diff(prediction, dim=-2)
        target_dz = torch.diff(target, dim=-2)
        mask_dz = mask[..., 1:, :] & mask[..., :-1, :]
        pred_dr = torch.diff(prediction, dim=-1)
        target_dr = torch.diff(target, dim=-1)
        mask_dr = mask[..., :, 1:] & mask[..., :, :-1]
        gradient_loss = 0.5 * (
            _masked_mean((pred_dz - target_dz) ** 2, mask_dz)
            + _masked_mean((pred_dr - target_dr) ** 2, mask_dr)
        )
    total = value_loss + convergence_weight * convergence_loss + gradient_weight * gradient_loss
    return total, {
        "value": float(value_loss.detach()),
        "convergence": float(convergence_loss.detach()),
        "gradient": float(gradient_loss.detach()),
    }


def _predict_batches(
    model: torch.nn.Module,
    features: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            outputs.append(model(features[start : start + batch_size].to(device)).cpu())
    return torch.cat(outputs)


def _benchmark_latency(
    model: torch.nn.Module,
    one_feature: np.ndarray,
    transform: TargetTransform,
    device: torch.device,
    *,
    repeats: int,
    groups: np.ndarray | None = None,
) -> dict[str, float]:
    model = model.to(device).eval()

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(10):
        tensor = torch.from_numpy(one_feature.copy()).to(device)
        with torch.no_grad():
            output = model(tensor)
            _ = transform.decode_tensor(output, groups).cpu().numpy()
        synchronize()
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        tensor = torch.from_numpy(one_feature.copy()).to(device)
        with torch.no_grad():
            output = model(tensor)
            _ = transform.decode_tensor(output, groups).cpu().numpy()
        synchronize()
        values.append((time.perf_counter() - started) * 1000.0)
    return {
        "repeats": repeats,
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "maximum_ms": float(np.max(values)),
    }


def run_experiment(
    mvp: MVPConfig,
    dataset_path: Path,
    experiment: dict,
    training_defaults: dict,
) -> Path:
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{experiment['id']}"
    run_dir = mvp.artifact_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(training_defaults)
    config.update(experiment.get("training", {}))
    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with np.load(dataset_path) as raw:
        targets_db = raw["tl_db"].astype(np.float32)
        masks = raw["valid_mask"].astype(bool)
        ssp_profiles = raw["ssp_speeds_mps"].astype(np.float32)
        ssp_depths = raw["ssp_depths_m"].astype(np.float32)
        depths = raw["depths_m"].astype(np.float32)
        ranges = raw["ranges_m"].astype(np.float32)
        bathymetry = (
            raw["bathymetry_depths_m"].astype(np.float32)
            if "bathymetry_depths_m" in raw
            else None
        )
        splits = raw["splits"].astype(str)
        sample_ids = raw["sample_ids"].astype(str)
        terrain_groups = (
            raw["bathymetry_profiles"].astype(str)
            if "bathymetry_profiles" in raw
            else np.full(len(targets_db), "flat")
        )
    grid_profiles = interpolate_ssp(ssp_depths, ssp_profiles, depths)
    features_np = build_features(
        grid_profiles,
        ranges,
        use_hankel=bool(experiment["model"].get("use_hankel_feature", False)),
        bathymetry_depths_m=bathymetry,
    )
    indices = {split: np.flatnonzero(splits == split) for split in ("train", "validation", "test")}
    target_transform_name = str(experiment.get("target_transform", "global_mean"))
    if target_transform_name == "terrain_mean":
        transform = TargetTransform.fit_grouped(
            targets_db[indices["train"]],
            masks[indices["train"]],
            terrain_groups[indices["train"]],
        )
        normalized_targets = transform.encode(targets_db, terrain_groups)[:, None]
    elif target_transform_name == "global_mean":
        transform = TargetTransform.fit(targets_db[indices["train"]], masks[indices["train"]])
        normalized_targets = transform.encode(targets_db)[:, None]
    else:
        raise ValueError(f"unknown target_transform {target_transform_name}")
    features = torch.from_numpy(features_np)
    targets = torch.from_numpy(normalized_targets)
    mask_tensor = torch.from_numpy(masks[:, None])

    train_index = indices["train"]
    loader = DataLoader(
        TensorDataset(features[train_index], targets[train_index], mask_tensor[train_index]),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=device.type == "cuda",
    )
    model = build_model(experiment["model"], features_np.shape[1]).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["max_epochs"]),
        eta_min=float(config["learning_rate"]) / 50.0,
    )
    validation_features = features[indices["validation"]]
    initial_validation_normalized = _predict_batches(
        model, validation_features, device, int(config["batch_size"])
    )
    initial_validation_db = transform.decode_tensor(
        initial_validation_normalized,
        terrain_groups[indices["validation"]]
        if transform.group_mean_fields_db is not None
        else None,
    ).numpy()
    best_rmse = tl_metrics(
        targets_db[indices["validation"]],
        initial_validation_db,
        masks[indices["validation"]],
    )["rmse_db"]
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    best_epoch = 0
    stale_epochs = 0
    history = []
    training_started = time.perf_counter()

    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        running_loss = 0.0
        for batch_features, batch_targets, batch_masks in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            batch_masks = batch_masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_features)
            loss, _ = _loss(prediction, batch_targets, batch_masks, experiment["loss"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.detach()) * len(batch_features)
        scheduler.step()

        validation_normalized = _predict_batches(
            model, validation_features, device, int(config["batch_size"])
        )
        validation_db = transform.decode_tensor(
            validation_normalized,
            terrain_groups[indices["validation"]]
            if transform.group_mean_fields_db is not None
            else None,
        ).numpy()
        validation_metric = tl_metrics(
            targets_db[indices["validation"]],
            validation_db,
            masks[indices["validation"]],
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / len(train_index),
                "validation_rmse_db": validation_metric["rmse_db"],
                "validation_mae_db": validation_metric["mae_db"],
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )
        if validation_metric["rmse_db"] < best_rmse:
            best_rmse = validation_metric["rmse_db"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{experiment['id']} epoch={epoch:03d} "
                f"train={history[-1]['train_loss']:.5f} val_rmse={validation_metric['rmse_db']:.4f}",
                flush=True,
            )
        if stale_epochs >= int(config["early_stopping_patience"]):
            break
    training_seconds = time.perf_counter() - training_started
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)

    prediction_normalized = _predict_batches(model, features, device, int(config["batch_size"]))
    prediction_db = transform.decode_tensor(
        prediction_normalized,
        terrain_groups if transform.group_mean_fields_db is not None else None,
    ).numpy()
    metrics = {}
    for split, split_index in indices.items():
        split_result = split_metrics(
            targets_db[split_index], prediction_db[split_index], masks[split_index]
        )
        difficult = high_gradient_mask(targets_db[split_index], masks[split_index], quantile=0.9)
        split_result["high_gradient"] = tl_metrics(
            targets_db[split_index], prediction_db[split_index], difficult
        )
        metrics[split] = split_result

    mean_baseline = np.broadcast_to(transform.mean_field_db, targets_db.shape)
    baseline_test = split_metrics(
        targets_db[indices["test"]],
        mean_baseline[indices["test"]],
        masks[indices["test"]],
    )["aggregate"]
    terrain_baseline_test = None
    if np.any(terrain_groups != "flat"):
        terrain_transform = TargetTransform.fit_grouped(
            targets_db[indices["train"]],
            masks[indices["train"]],
            terrain_groups[indices["train"]],
        )
        terrain_mean = terrain_transform._means(terrain_groups)
        terrain_baseline_test = split_metrics(
            targets_db[indices["test"]],
            terrain_mean[indices["test"]],
            masks[indices["test"]],
        )["aggregate"]
    latency = {
        "gpu": _benchmark_latency(
            model,
            features_np[indices["test"][:1]],
            transform,
            torch.device("cuda"),
            repeats=200,
            groups=(
                terrain_groups[indices["test"][:1]]
                if transform.group_mean_fields_db is not None
                else None
            ),
        )
        if torch.cuda.is_available()
        else None,
    }
    # CPU timing is diagnostic; keep repetitions low for the larger variants.
    cpu_model = copy.deepcopy(model).cpu()
    latency["cpu"] = _benchmark_latency(
        cpu_model,
        features_np[indices["test"][:1]],
        transform,
        torch.device("cpu"),
        repeats=20,
        groups=(
            terrain_groups[indices["test"][:1]]
            if transform.group_mean_fields_db is not None
            else None
        ),
    )
    model.to(device)

    test_metric = metrics["test"]["aggregate"]
    latency_metric = latency["gpu"] or latency["cpu"]
    acceptance = {
        "rmse_pass": test_metric["rmse_db"] <= mvp.acceptance.maximum_test_rmse_db,
        "mae_pass": test_metric["mae_db"] <= mvp.acceptance.maximum_test_mae_db,
        "latency_pass": latency_metric["p95_ms"] <= mvp.acceptance.maximum_p95_latency_ms,
    }
    minimum_baseline = experiment.get("minimum_mean_baseline_test_rmse_db")
    if minimum_baseline is not None:
        acceptance["mean_baseline_informative_pass"] = (
            baseline_test["rmse_db"] > float(minimum_baseline)
        )
    acceptance["overall_pass"] = all(acceptance.values())
    result = {
        "run_id": run_id,
        "experiment_id": experiment["id"],
        "hypothesis": experiment.get("hypothesis", ""),
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": __import__("hashlib").sha256(dataset_path.read_bytes()).hexdigest(),
        "model_config": experiment["model"],
        "loss_config": experiment["loss"],
        "training_config": config,
        "device": str(device),
        "hardware": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu_threads": torch.get_num_threads(),
        },
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "training_seconds": training_seconds,
        "target_transform": {
            "residual_scale_db": transform.residual_scale_db,
            "name": target_transform_name,
        },
        "mean_field_baseline_test": baseline_test,
        "terrain_mean_baseline_test": terrain_baseline_test,
        "baseline_improvement_rmse_db": baseline_test["rmse_db"] - test_metric["rmse_db"],
        "baseline_rmse_reduction_percent": 100.0
        * (baseline_test["rmse_db"] - test_metric["rmse_db"])
        / baseline_test["rmse_db"],
        "metrics": metrics,
        "latency": latency,
        "acceptance": acceptance,
        "sample_ids": {
            split: sample_ids[split_index].tolist() for split, split_index in indices.items()
        },
    }
    checkpoint = {
        "model_state": best_state,
        "model_config": experiment["model"],
        "in_channels": features_np.shape[1],
        "target_transform": transform.state_dict(),
        "ranges_m": ranges,
        "depths_m": depths,
        "ssp_depths_m": ssp_depths,
        "bathymetry_depths_m": bathymetry,
        "bathymetry_profiles": terrain_groups,
        "experiment_id": experiment["id"],
    }
    torch.save(checkpoint, run_dir / "model.pt")
    np.savez_compressed(
        run_dir / "predictions.npz",
        prediction_tl_db=prediction_db,
        reference_tl_db=targets_db,
        valid_mask=masks,
        sample_ids=sample_ids,
        splits=splits,
        ranges_m=ranges,
        depths_m=depths,
        ssp_speeds_mps=ssp_profiles,
        ssp_depths_m=ssp_depths,
        bathymetry_profiles=terrain_groups,
        **({"bathymetry_depths_m": bathymetry} if bathymetry is not None else {}),
    )
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return run_dir
