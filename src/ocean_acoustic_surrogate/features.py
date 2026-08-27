"""Input features and training-only target normalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import hankel1


def interpolate_ssp(
    ssp_depths_m: np.ndarray, ssp_speeds_mps: np.ndarray, output_depths_m: np.ndarray
) -> np.ndarray:
    return np.stack(
        [np.interp(output_depths_m, ssp_depths_m, profile) for profile in ssp_speeds_mps]
    ).astype(np.float32)


def hankel_feature(ranges_m: np.ndarray, frequency_hz: float = 1000.0) -> np.ndarray:
    """Stable normalized log-amplitude of the cylindrical Hankel Green function."""
    wavenumber = 2 * np.pi * frequency_hz / 1500.0
    amplitude = np.abs(hankel1(0, wavenumber * np.maximum(ranges_m, 1.0)))
    feature = np.log10(np.maximum(amplitude, 1e-12))
    feature = (feature - feature.mean()) / max(feature.std(), 1e-6)
    return feature.astype(np.float32)


def build_features(
    profiles_on_grid: np.ndarray,
    ranges_m: np.ndarray,
    *,
    use_hankel: bool,
    bathymetry_depths_m: np.ndarray | None = None,
) -> np.ndarray:
    n_samples, n_depths = profiles_on_grid.shape
    n_ranges = len(ranges_m)
    sound_speed = ((profiles_on_grid - 1500.0) / 50.0)[:, None, :, None]
    sound_speed = np.broadcast_to(sound_speed, (n_samples, 1, n_depths, n_ranges))
    channels = [sound_speed]
    if bathymetry_depths_m is not None:
        bottom = np.asarray(bathymetry_depths_m, dtype=np.float32)
        if bottom.shape == (n_ranges,):
            bottom = np.broadcast_to(bottom[None, :], (n_samples, n_ranges))
        if bottom.shape != (n_samples, n_ranges):
            raise ValueError("bathymetry_depths_m must be [range] or [sample, range]")
        terrain = (bottom / 2000.0)[:, None, None, :]
        channels.append(np.broadcast_to(terrain, (n_samples, 1, n_depths, n_ranges)))
    if use_hankel:
        hankel = hankel_feature(ranges_m)[None, None, None, :]
        hankel = np.broadcast_to(hankel, (n_samples, 1, n_depths, n_ranges))
        channels.append(hankel)
    return np.concatenate(channels, axis=1).astype(np.float32, copy=True)


@dataclass
class TargetTransform:
    mean_field_db: np.ndarray
    residual_scale_db: float
    group_mean_fields_db: dict[str, np.ndarray] | None = None

    @classmethod
    def fit(cls, targets_db: np.ndarray, masks: np.ndarray) -> TargetTransform:
        valid_count = masks.sum(axis=0)
        total = np.where(masks, targets_db, 0.0).sum(axis=0)
        global_mean = float(targets_db[masks].mean())
        mean_field = np.divide(
            total,
            valid_count,
            out=np.full_like(total, global_mean, dtype=np.float64),
            where=valid_count > 0,
        ).astype(np.float32)
        residual = targets_db - mean_field[None]
        scale = float(np.std(residual[masks]))
        return cls(mean_field_db=mean_field, residual_scale_db=max(scale, 1.0))

    @classmethod
    def fit_grouped(
        cls,
        targets_db: np.ndarray,
        masks: np.ndarray,
        groups: np.ndarray,
    ) -> TargetTransform:
        global_transform = cls.fit(targets_db, masks)
        group_means = {}
        residuals = []
        for group in np.unique(groups.astype(str)):
            selected = groups.astype(str) == group
            local = cls.fit(targets_db[selected], masks[selected])
            group_means[str(group)] = local.mean_field_db
            residuals.append(
                (targets_db[selected] - local.mean_field_db[None])[masks[selected]]
            )
        scale = max(float(np.std(np.concatenate(residuals))), 1.0)
        return cls(
            mean_field_db=global_transform.mean_field_db,
            residual_scale_db=scale,
            group_mean_fields_db=group_means,
        )

    def _means(self, groups: np.ndarray | None) -> np.ndarray:
        if self.group_mean_fields_db is None:
            if groups is None:
                return self.mean_field_db[None]
            return np.broadcast_to(self.mean_field_db, (len(groups), *self.mean_field_db.shape))
        if groups is None:
            raise ValueError("group labels are required for a grouped target transform")
        return np.stack([self.group_mean_fields_db[str(group)] for group in groups])

    def encode(self, targets_db: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
        means = self._means(groups)
        return ((targets_db - means) / self.residual_scale_db).astype(np.float32)

    def decode_tensor(
        self, normalized: torch.Tensor, groups: np.ndarray | None = None
    ) -> torch.Tensor:
        mean = torch.as_tensor(self._means(groups), dtype=normalized.dtype, device=normalized.device)
        return normalized[:, 0] * self.residual_scale_db + mean

    def state_dict(self) -> dict:
        return {
            "mean_field_db": self.mean_field_db,
            "residual_scale_db": self.residual_scale_db,
            "group_mean_fields_db": self.group_mean_fields_db,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> TargetTransform:
        groups = state.get("group_mean_fields_db")
        return cls(
            mean_field_db=np.asarray(state["mean_field_db"]),
            residual_scale_db=float(state["residual_scale_db"]),
            group_mean_fields_db=(
                {str(key): np.asarray(value) for key, value in groups.items()}
                if groups is not None
                else None
            ),
        )
