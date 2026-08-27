"""Independent checkpoint reload and sealed-test verification."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .config import MVPConfig
from .features import TargetTransform, build_features, interpolate_ssp
from .metrics import split_metrics
from .models import build_model
from .training import _benchmark_latency, _predict_batches


def verify_run(
    mvp: MVPConfig,
    dataset_path: Path,
    run_dir: Path,
    device_name: str = "auto",
) -> Path:
    """Reload a saved model in a fresh process and re-score only the frozen test split."""
    recorded = json.loads((run_dir / "metrics.json").read_text())
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    with np.load(dataset_path) as raw:
        targets = raw["tl_db"].astype(np.float32)
        masks = raw["valid_mask"].astype(bool)
        profiles = raw["ssp_speeds_mps"].astype(np.float32)
        ssp_depths = raw["ssp_depths_m"].astype(np.float32)
        depths = raw["depths_m"].astype(np.float32)
        ranges = raw["ranges_m"].astype(np.float32)
        bathymetry = (
            raw["bathymetry_depths_m"].astype(np.float32)
            if "bathymetry_depths_m" in raw
            else None
        )
        splits = raw["splits"].astype(str)
    test = np.flatnonzero(splits == "test")
    features_np = build_features(
        interpolate_ssp(ssp_depths, profiles, depths),
        ranges,
        use_hankel=bool(checkpoint["model_config"].get("use_hankel_feature", False)),
        bathymetry_depths_m=bathymetry,
    )
    transform = TargetTransform(
        mean_field_db=np.asarray(checkpoint["target_transform"]["mean_field_db"]),
        residual_scale_db=float(checkpoint["target_transform"]["residual_scale_db"]),
    )
    model = build_model(checkpoint["model_config"], int(checkpoint["in_channels"]))
    model.load_state_dict(checkpoint["model_state"])
    if device_name not in {"auto", "cpu", "cuda"}:
        raise ValueError("device_name must be auto, cpu, or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA verification requested but CUDA is unavailable")
    resolved_device = "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    if resolved_device == "auto":
        resolved_device = "cpu"
    device = torch.device(resolved_device)
    model.to(device).eval()
    prediction = transform.decode_tensor(
        _predict_batches(
            model,
            torch.from_numpy(features_np[test]),
            device,
            int(recorded["training_config"]["batch_size"]),
        )
    ).numpy()
    metrics = split_metrics(targets[test], prediction, masks[test])
    latency = _benchmark_latency(
        model,
        features_np[test[:1]],
        transform,
        device,
        repeats=200,
    )
    original_rmse = float(recorded["metrics"]["test"]["aggregate"]["rmse_db"])
    recomputed_rmse = float(metrics["aggregate"]["rmse_db"])
    rmse_delta = abs(recomputed_rmse - original_rmse)
    rmse_tolerance = 1e-4
    recorded_device = str(recorded["device"])
    same_device = recorded_device.startswith(device.type)
    reload_matches = rmse_delta <= rmse_tolerance if same_device else None
    result = {
        "verified_at": datetime.now(UTC).isoformat(),
        "run_id": recorded["run_id"],
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "dataset_hash_matches_run": dataset_sha256 == recorded["dataset_sha256"],
        "recorded_device": recorded_device,
        "verification_device": str(device),
        "same_device_as_recorded": same_device,
        "recorded_test_rmse_db": original_rmse,
        "recomputed_test_rmse_db": recomputed_rmse,
        "rmse_absolute_delta_db": rmse_delta,
        "rmse_match_tolerance_db": rmse_tolerance,
        "checkpoint_reload_matches_rmse": reload_matches,
        "test_metrics": metrics,
        "latency": {"device": str(device), **latency},
    }
    result["acceptance"] = {
        "rmse_pass": metrics["aggregate"]["rmse_db"]
        <= mvp.acceptance.maximum_test_rmse_db,
        "mae_pass": metrics["aggregate"]["mae_db"] <= mvp.acceptance.maximum_test_mae_db,
        "latency_pass": latency["p95_ms"] <= mvp.acceptance.maximum_p95_latency_ms,
    }
    result["acceptance"]["overall_pass"] = all(result["acceptance"].values())
    result["verification_pass"] = (
        result["dataset_hash_matches_run"]
        and result["acceptance"]["overall_pass"]
        and (not same_device or bool(reload_matches))
    )
    output = run_dir / f"independent_verification_{device.type}.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return output
