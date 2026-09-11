"""Generate one non-overlapping shard of a resumable Bellhop dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from ocean_acoustic_surrogate.config import MVPConfig
from ocean_acoustic_surrogate.dataset import (
    _bathymetry_for_record,
    _dataset_splits,
    _run_task,
    _source_depth_for_record,
    build_task,
)
from ocean_acoustic_surrogate.ssp import PARAMETER_NAMES, build_ssp_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    args = parser.parse_args()

    config = MVPConfig.from_yaml(args.config)
    if not 0 <= args.start < args.stop <= args.samples:
        raise ValueError("require 0 <= start < stop <= samples")
    root = config.dataset_root / f"n{args.samples}"
    samples_root = root / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)
    os.environ["OUTPUT_DIR"] = str(root / "bellhop_cases")
    terrain_count = (
        len(config.contract.bathymetry.profiles) if config.contract.bathymetry is not None else 1
    )
    records = build_ssp_records(
        config.ssp_family,
        args.samples,
        config.contract.seed,
        template_cycle_stride=terrain_count,
    )
    splits = _dataset_splits(config, args.samples)
    failures = []
    completed = 0
    for index in range(args.start, args.stop):
        record = records[index]
        sample_dir = samples_root / record.sample_id
        array_path = sample_dir / "sample.npz"
        metadata_path = sample_dir / "metadata.json"
        if array_path.exists() and metadata_path.exists():
            completed += 1
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
            bathymetry = _bathymetry_for_record(config, record)
            bottom = (
                np.interp(ranges, bathymetry.ranges_m, bathymetry.depths_m).astype(np.float32)
                if bathymetry is not None
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
                **({"bathymetry_depths_m": bottom} if bottom is not None else {}),
            )
            metadata = {
                "sample_id": record.sample_id,
                "split": str(splits[index]),
                "parameters": dict(zip(PARAMETER_NAMES, map(float, record.parameters))),
                "bellhop_wall_seconds": wall_seconds,
                "bellhop_case_dir": str(case_dir),
                "field_mode": config.contract.field_mode,
                "num_rays": config.contract.reference_num_rays,
                "finite_coverage": float(valid.mean()),
                "config_hash": config.config_hash,
                "bathymetry_profile": bathymetry.name if bathymetry is not None else "flat",
                "ssp_profile": record.profile_name,
                "source_depth_m": _source_depth_for_record(config, record),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            completed += 1
        except Exception as exc:  # noqa: BLE001 - report the complete shard
            failures.append({"sample_id": record.sample_id, "error": str(exc)})
        if (index - args.start + 1) % 16 == 0 or index + 1 == args.stop:
            print(
                f"shard {args.start}:{args.stop} progress={index + 1 - args.start}/"
                f"{args.stop - args.start} completed={completed} failures={len(failures)}",
                flush=True,
            )
    if failures:
        raise RuntimeError(json.dumps(failures, indent=2))


if __name__ == "__main__":
    main()
