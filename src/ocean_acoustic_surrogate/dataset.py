"""Resumable Bellhop label generation and dataset packaging."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import platform
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from ocean_acoustic_agent import SimulationTask, run_simulation
from ocean_acoustic_agent.schemas import (
    BathymetryPoint,
    BathymetryProfile,
    EnvironmentSpec,
    SeabedType,
    SoundSpeedPoint,
    SoundSpeedProfile,
)
from ocean_acoustic_agent.schemas.task import ReceiverGrid

from .config import BathymetryProfileConfig, MVPConfig
from .ssp import PARAMETER_NAMES, SSPRecord, assign_splits, build_ssp_records


def _git_sha(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bathymetry_for_record(
    config: MVPConfig, record: SSPRecord
) -> BathymetryProfileConfig | None:
    family = config.contract.bathymetry
    if family is None:
        return None
    sample_index = int(record.sample_id.rsplit("_", maxsplit=1)[-1])
    return family.profiles[sample_index % len(family.profiles)]


def _dataset_splits(config: MVPConfig, n_samples: int) -> np.ndarray:
    family = config.contract.bathymetry
    ssp_profile_count = max(1, len(config.ssp_family.profiles))
    if family is None and ssp_profile_count == 1:
        return assign_splits(
            n_samples,
            config.contract.seed,
            config.split.train_fraction,
            config.split.validation_fraction,
        )
    terrain_count = len(family.profiles) if family is not None else 1
    group_count = terrain_count * ssp_profile_count
    splits = np.empty(n_samples, dtype="U10")
    for group_index in range(group_count):
        indices = np.arange(group_index, n_samples, group_count)
        local = assign_splits(
            len(indices),
            config.contract.seed + group_index,
            config.split.train_fraction,
            config.split.validation_fraction,
        )
        splits[indices] = local
    return splits


def build_task(config: MVPConfig, record: SSPRecord, num_rays: int, task_id: str) -> SimulationTask:
    contract = config.contract
    seabed = contract.seabed
    bathymetry = None
    selected_bathymetry = _bathymetry_for_record(config, record)
    if selected_bathymetry is not None:
        bathymetry = BathymetryProfile(
            points=[
                BathymetryPoint(range_m=float(distance), depth_m=float(depth))
                for distance, depth in zip(
                    selected_bathymetry.ranges_m, selected_bathymetry.depths_m
                )
            ]
        )
    environment = EnvironmentSpec(
        water_depth_m=contract.water_depth_m,
        bathymetry=bathymetry,
        ssp=SoundSpeedProfile(
            points=[
                SoundSpeedPoint(depth_m=float(depth), speed_mps=float(speed))
                for depth, speed in zip(record.depths_m, record.speeds_mps)
            ],
            interpolation="linear",
        ),
        seabed=SeabedType(
            kind=seabed.kind,
            speed_mps=seabed.speed_mps,
            density_kgm3=seabed.density_kgm3,
            attenuation_dbperlambda=seabed.attenuation_dbperlambda,
        ),
    )
    return SimulationTask(
        task_id=task_id,
        model_name="bellhop",
        frequency_hz=contract.frequency_hz,
        source_depth_m=contract.source_depth_m,
        receiver_depth_m=1000.0,
        receiver_range_m=contract.range_end_m,
        num_rays=num_rays,
        environment=environment,
        receiver_grid=ReceiverGrid(
            range_start_m=contract.range_start_m,
            range_end_m=contract.range_end_m,
            range_steps=contract.range_steps,
            depth_start_m=contract.depth_start_m,
            depth_end_m=contract.depth_end_m,
            depth_steps=contract.depth_steps,
        ),
        runtime_options={"field_mode": contract.field_mode},
        metadata={
            "dataset_id": contract.dataset_id,
            "config_hash": config.config_hash,
            "ssp_family": config.ssp_family.name,
            "parameter_names": list(PARAMETER_NAMES),
            "parameters": record.parameters.tolist(),
            "bathymetry_profile": (
                selected_bathymetry.name if selected_bathymetry is not None else "flat"
            ),
        },
    )


def _run_task(task: SimulationTask) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, Path]:
    started = time.perf_counter()
    result = run_simulation(task)
    wall_seconds = time.perf_counter() - started
    if result.status == "failed" or result.tl_field is None:
        raise RuntimeError(result.error_message or f"Bellhop failed for {task.task_id}")
    tl = np.asarray(result.tl_field["tl_db"], dtype=np.float32)
    ranges = np.asarray(result.tl_field["ranges_m"], dtype=np.float32)
    depths = np.asarray(result.tl_field["depths_m"], dtype=np.float32)
    if tl.shape != (len(depths), len(ranges)):
        raise ValueError(f"unexpected TL shape {tl.shape}")
    if result.case_dir is None:
        raise RuntimeError("Bellhop result did not preserve its case directory")
    return tl, ranges, depths, wall_seconds, result.case_dir


def run_pilot(config: MVPConfig, n_samples: int = 8) -> Path:
    """Audit label convergence across the frozen ray-count ladder."""
    root = config.dataset_root / "pilot"
    root.mkdir(parents=True, exist_ok=True)
    os.environ["OUTPUT_DIR"] = str(root / "bellhop_cases")
    terrain_count = (
        len(config.contract.bathymetry.profiles) if config.contract.bathymetry is not None else 1
    )
    records = build_ssp_records(
        config.ssp_family,
        n_samples,
        config.contract.seed + 100_000,
        template_cycle_stride=terrain_count,
    )
    comparisons: dict[str, list[dict[str, float]]] = {}
    timings: dict[str, list[float]] = {str(rays): [] for rays in config.contract.pilot_ray_counts}
    failures = []

    for record in records:
        fields = []
        sample_dir = root / record.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        for rays in config.contract.pilot_ray_counts:
            cache = sample_dir / f"rays_{rays}.npz"
            if cache.exists():
                with np.load(cache) as raw:
                    tl = raw["tl_db"]
                    ranges = raw["ranges_m"]
                    depths = raw["depths_m"]
                    wall_seconds = float(raw["wall_seconds"])
            else:
                try:
                    task = build_task(config, record, rays, f"pilot_{record.sample_id}_r{rays}")
                    tl, ranges, depths, wall_seconds, _ = _run_task(task)
                    np.savez_compressed(
                        cache,
                        tl_db=tl,
                        ranges_m=ranges,
                        depths_m=depths,
                        wall_seconds=np.float64(wall_seconds),
                    )
                except Exception as exc:  # noqa: BLE001 - keep the audit resumable
                    failures.append(
                        {"sample_id": record.sample_id, "rays": rays, "error": str(exc)}
                    )
                    break
            timings[str(rays)].append(wall_seconds)
            fields.append((rays, tl))
        sample_comparisons = []
        for (lower_rays, lower), (upper_rays, upper) in itertools.pairwise(fields):
            mask = np.isfinite(lower) & np.isfinite(upper)
            error = lower[mask].astype(np.float64) - upper[mask].astype(np.float64)
            sample_comparisons.append(
                {
                    "lower_rays": lower_rays,
                    "upper_rays": upper_rays,
                    "mae_db": float(np.mean(np.abs(error))),
                    "rmse_db": float(np.sqrt(np.mean(error**2))),
                    "p95_absolute_error_db": float(np.percentile(np.abs(error), 95)),
                    "common_coverage": float(mask.mean()),
                }
            )
        comparisons[record.sample_id] = sample_comparisons

    aggregate = []
    for lower, upper in zip(config.contract.pilot_ray_counts, config.contract.pilot_ray_counts[1:]):
        items = [
            item
            for values in comparisons.values()
            for item in values
            if item["lower_rays"] == lower and item["upper_rays"] == upper
        ]
        if items:
            aggregate.append(
                {
                    "lower_rays": lower,
                    "upper_rays": upper,
                    "n_samples": len(items),
                    "mean_mae_db": float(np.mean([item["mae_db"] for item in items])),
                    "mean_rmse_db": float(np.mean([item["rmse_db"] for item in items])),
                    "worst_rmse_db": float(np.max([item["rmse_db"] for item in items])),
                    "mean_p95_absolute_error_db": float(
                        np.mean([item["p95_absolute_error_db"] for item in items])
                    ),
                }
            )
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_id": config.contract.dataset_id,
        "config_hash": config.config_hash,
        "field_mode": config.contract.field_mode,
        "n_samples": n_samples,
        "ray_counts": config.contract.pilot_ray_counts,
        "aggregate": aggregate,
        "per_sample": comparisons,
        "timing_seconds": {
            rays: {
                "median": float(np.median(values)) if values else None,
                "p95": float(np.percentile(values, 95)) if values else None,
            }
            for rays, values in timings.items()
        },
        "failures": failures,
    }
    path = root / "convergence_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return path


def _reuse_prefix_labels(
    config: MVPConfig,
    records: list[SSPRecord],
    splits: np.ndarray,
    target_samples_root: Path,
    source_root: Path,
) -> int:
    """Reuse a numerically identical prefix from a previously frozen dataset."""
    if source_root.name == "dataset.npz":
        source_root = source_root.parent
    source_samples_root = source_root / "samples"
    if not source_samples_root.is_dir():
        raise FileNotFoundError(f"reuse source has no samples directory: {source_samples_root}")

    reused = 0
    for index, record in enumerate(records):
        source_sample = source_samples_root / record.sample_id
        source_array = source_sample / "sample.npz"
        source_metadata = source_sample / "metadata.json"
        if not source_array.exists() or not source_metadata.exists():
            continue
        target_sample = target_samples_root / record.sample_id
        target_array = target_sample / "sample.npz"
        target_metadata = target_sample / "metadata.json"
        if target_array.exists() and target_metadata.exists():
            continue

        metadata = json.loads(source_metadata.read_text())
        selected_bathymetry = _bathymetry_for_record(config, record)
        expected_terrain = selected_bathymetry.name if selected_bathymetry is not None else "flat"
        with np.load(source_array) as raw:
            if not np.array_equal(raw["parameters"], record.parameters):
                raise ValueError(
                    f"reuse prefix parameters differ for {record.sample_id}; refusing stale label"
                )
            if not np.array_equal(raw["ssp_speeds_mps"], record.speeds_mps):
                raise ValueError(
                    f"reuse prefix SSP differs for {record.sample_id}; refusing stale label"
                )
        if metadata.get("bathymetry_profile") != expected_terrain:
            raise ValueError(
                f"reuse prefix terrain differs for {record.sample_id}; refusing stale label"
            )
        if int(metadata.get("num_rays", -1)) != config.contract.reference_num_rays:
            raise ValueError(
                f"reuse prefix ray count differs for {record.sample_id}; refusing stale label"
            )

        target_sample.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source_array, target_array)
        except OSError:
            shutil.copy2(source_array, target_array)
        metadata.update(
            {
                "split": str(splits[index]),
                "config_hash": config.config_hash,
                "reused_from": str(source_sample),
            }
        )
        target_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
        reused += 1
    return reused


def generate_dataset(
    config: MVPConfig,
    n_samples: int,
    reuse_prefix_from: Path | None = None,
) -> Path:
    """Generate or resume a high-quality Bellhop dataset and package it."""
    root = config.dataset_root / f"n{n_samples}"
    samples_root = root / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)
    os.environ["OUTPUT_DIR"] = str(root / "bellhop_cases")

    terrain_count = (
        len(config.contract.bathymetry.profiles) if config.contract.bathymetry is not None else 1
    )
    records = build_ssp_records(
        config.ssp_family,
        n_samples,
        config.contract.seed,
        template_cycle_stride=terrain_count,
    )
    splits = _dataset_splits(config, n_samples)
    reused_count = 0
    if reuse_prefix_from is not None:
        reused_count = _reuse_prefix_labels(
            config,
            records,
            splits,
            samples_root,
            reuse_prefix_from,
        )
        print(f"reused {reused_count}/{n_samples} frozen prefix labels", flush=True)
    failures = []
    for index, (record, split) in enumerate(zip(records, splits)):
        sample_dir = samples_root / record.sample_id
        array_path = sample_dir / "sample.npz"
        metadata_path = sample_dir / "metadata.json"
        if array_path.exists() and metadata_path.exists():
            continue
        sample_dir.mkdir(parents=True, exist_ok=True)
        try:
            task = build_task(
                config,
                record,
                config.contract.reference_num_rays,
                f"{config.contract.dataset_id}_{record.sample_id}",
            )
            tl, ranges, depths, wall_seconds, case_dir = _run_task(task)
            valid = np.isfinite(tl)
            scored = np.where(valid, tl, config.contract.invalid_tl_fill_db).astype(np.float32)
            selected_bathymetry = _bathymetry_for_record(config, record)
            bathymetry_on_grid = (
                np.interp(
                    ranges,
                    selected_bathymetry.ranges_m,
                    selected_bathymetry.depths_m,
                ).astype(np.float32)
                if selected_bathymetry is not None
                else None
            )
            np.savez_compressed(
                array_path,
                tl_db=scored,
                valid_mask=valid,
                ranges_m=ranges,
                depths_m=depths,
                ssp_depths_m=record.depths_m,
                ssp_speeds_mps=record.speeds_mps,
                parameters=record.parameters,
                **(
                    {"bathymetry_depths_m": bathymetry_on_grid}
                    if bathymetry_on_grid is not None
                    else {}
                ),
            )
            metadata = {
                "sample_id": record.sample_id,
                "split": str(split),
                "parameters": dict(zip(PARAMETER_NAMES, map(float, record.parameters))),
                "bellhop_wall_seconds": wall_seconds,
                "bellhop_case_dir": str(case_dir),
                "field_mode": config.contract.field_mode,
                "num_rays": config.contract.reference_num_rays,
                "finite_coverage": float(valid.mean()),
                "config_hash": config.config_hash,
                "bathymetry_profile": (
                    selected_bathymetry.name if selected_bathymetry is not None else "flat"
                ),
                "ssp_profile": record.profile_name,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        except Exception as exc:  # noqa: BLE001 - record failure and finish manifest
            failures.append({"sample_id": record.sample_id, "error": str(exc)})
        if (index + 1) % 16 == 0 or index + 1 == n_samples:
            print(f"generated {index + 1}/{n_samples}; failures={len(failures)}", flush=True)

    arrays: dict[str, list[np.ndarray]] = {
        "tl_db": [],
        "valid_mask": [],
        "ssp_speeds_mps": [],
        "parameters": [],
    }
    if config.contract.bathymetry is not None:
        arrays["bathymetry_depths_m"] = []
    metadata_records = []
    ranges = depths = ssp_depths = None
    for record in records:
        sample_dir = samples_root / record.sample_id
        metadata_path = sample_dir / "metadata.json"
        array_path = sample_dir / "sample.npz"
        if not metadata_path.exists() or not array_path.exists():
            continue
        metadata_records.append(json.loads(metadata_path.read_text()))
        with np.load(array_path) as raw:
            for key, values in arrays.items():
                values.append(raw[key].copy())
            if ranges is None:
                ranges = raw["ranges_m"].copy()
                depths = raw["depths_m"].copy()
                ssp_depths = raw["ssp_depths_m"].copy()
    if len(metadata_records) != n_samples:
        raise RuntimeError(f"only {len(metadata_records)}/{n_samples} samples succeeded")

    dataset_path = root / "dataset.npz"
    np.savez_compressed(
        dataset_path,
        **{key: np.stack(values) for key, values in arrays.items()},
        ranges_m=ranges,
        depths_m=depths,
        ssp_depths_m=ssp_depths,
        sample_ids=np.asarray([record["sample_id"] for record in metadata_records]),
        splits=np.asarray([record["split"] for record in metadata_records]),
        bathymetry_profiles=np.asarray(
            [record["bathymetry_profile"] for record in metadata_records]
        ),
        ssp_profiles=np.asarray([record["ssp_profile"] for record in metadata_records]),
        **({"bathymetry_ranges_m": ranges} if config.contract.bathymetry is not None else {}),
    )
    project_root = Path(__file__).resolve().parents[2]
    acoustic_root = project_root.parent / "ocean-acoustic-agent"
    manifest: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_id": config.contract.dataset_id,
        "n_samples": n_samples,
        "split_counts": {
            name: sum(record["split"] == name for record in metadata_records)
            for name in ("train", "validation", "test")
        },
        "shape": list(arrays["tl_db"][0].shape),
        "config": config.canonical_dict(),
        "config_hash": config.config_hash,
        "dataset_sha256": _sha256(dataset_path),
        "project_git_sha": _git_sha(project_root),
        "ocean_acoustic_agent_git_sha": _git_sha(acoustic_root),
        "python": platform.python_version(),
        "bellhop_wall_seconds": {
            "total": float(sum(record["bellhop_wall_seconds"] for record in metadata_records)),
            "median": float(np.median([r["bellhop_wall_seconds"] for r in metadata_records])),
            "p95": float(np.percentile([r["bellhop_wall_seconds"] for r in metadata_records], 95)),
        },
        "finite_coverage": float(
            np.mean([record["finite_coverage"] for record in metadata_records])
        ),
        "label_provenance": {
            "reused_prefix_count": int(
                sum("reused_from" in record for record in metadata_records)
            ),
            "generated_in_dataset_count": int(
                sum("reused_from" not in record for record in metadata_records)
            ),
            "reuse_source": str(reuse_prefix_from) if reuse_prefix_from is not None else None,
        },
        "failures": failures,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return dataset_path
