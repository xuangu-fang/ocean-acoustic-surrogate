"""Deterministic low-dimensional synthetic June deep-ocean SSP family."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import qmc

from .config import SSPFamilyConfig

PARAMETER_NAMES = (
    "global_offset_mps",
    "thermocline_amplitude_mps",
    "channel_axis_shift_m",
    "deep_gradient_mps",
)


@dataclass(frozen=True)
class SSPRecord:
    sample_id: str
    parameters: np.ndarray
    depths_m: np.ndarray
    speeds_mps: np.ndarray
    profile_name: str = "base"


def latin_hypercube_parameters(config: SSPFamilyConfig, n_samples: int, seed: int) -> np.ndarray:
    """Generate a reproducible space-filling design inside the narrow family."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    sampler = qmc.LatinHypercube(d=4, seed=seed, optimization="random-cd")
    unit = sampler.random(n_samples)
    bounds = np.asarray(
        [
            config.global_offset_mps,
            config.thermocline_amplitude_mps,
            config.channel_axis_shift_m,
            config.deep_gradient_mps,
        ],
        dtype=np.float64,
    )
    return qmc.scale(unit, bounds[:, 0], bounds[:, 1]).astype(np.float32)


def profile_from_parameters(
    config: SSPFamilyConfig,
    parameters: np.ndarray,
    base_speeds_mps: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct one smooth, bounded SSP from four interpretable perturbations."""
    global_offset, thermocline_amplitude, axis_shift, deep_gradient = map(float, parameters)
    base_depths = np.asarray(config.depths_m, dtype=np.float64)
    base_speeds = np.asarray(
        config.base_speeds_mps if base_speeds_mps is None else base_speeds_mps,
        dtype=np.float64,
    )
    depths = np.linspace(base_depths[0], base_depths[-1], config.interpolation_points)

    # Shifting the profile vertically moves the sound-channel axis without
    # inventing pixel-scale structure.  Boundary extrapolation is clamped.
    shifted_depths = np.clip(depths - axis_shift, base_depths[0], base_depths[-1])
    base = CubicSpline(base_depths, base_speeds, bc_type="natural")(shifted_depths)
    thermocline = thermocline_amplitude * np.exp(-0.5 * ((depths - 250.0) / 220.0) ** 2)
    deep = deep_gradient * np.clip((depths - 1000.0) / 1000.0, 0.0, 1.0)
    speeds = base + global_offset + thermocline + deep
    if not np.all((1475.0 <= speeds) & (speeds <= 1565.0)):
        raise ValueError("sampled SSP left the frozen physical bounds")
    return depths.astype(np.float32), speeds.astype(np.float32)


def build_ssp_records(
    config: SSPFamilyConfig,
    n_samples: int,
    seed: int,
    template_cycle_stride: int = 1,
) -> list[SSPRecord]:
    parameters = latin_hypercube_parameters(config, n_samples, seed)
    records = []
    for index, values in enumerate(parameters):
        profile = (
            config.profiles[(index // template_cycle_stride) % len(config.profiles)]
            if config.profiles
            else None
        )
        depths, speeds = profile_from_parameters(
            config,
            values,
            np.asarray(profile.speeds_mps) if profile is not None else None,
        )
        records.append(
            SSPRecord(
                sample_id=f"ssp_{index:05d}",
                parameters=values,
                depths_m=depths,
                speeds_mps=speeds,
                profile_name=profile.name if profile is not None else "base",
            )
        )
    return records


def assign_splits(n_samples: int, seed: int, train: float, validation: float) -> np.ndarray:
    """Assign deterministic IID splits without changing the sampled design."""
    rng = np.random.default_rng(seed + 17)
    order = rng.permutation(n_samples)
    n_train = round(n_samples * train)
    n_validation = round(n_samples * validation)
    splits = np.full(n_samples, "test", dtype="U10")
    splits[order[:n_train]] = "train"
    splits[order[n_train : n_train + n_validation]] = "validation"
    return splits
