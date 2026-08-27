"""Frozen TL error metrics."""

from __future__ import annotations

import numpy as np


def tl_metrics(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(reference) & np.isfinite(prediction)
    if not np.any(mask):
        raise ValueError("metric mask contains no valid cells")
    error = prediction[mask] - reference[mask]
    return {
        "rmse_db": float(np.sqrt(np.mean(error**2))),
        "mae_db": float(np.mean(np.abs(error))),
        "bias_db": float(np.mean(error)),
        "p95_absolute_error_db": float(np.percentile(np.abs(error), 95)),
        "maximum_absolute_error_db": float(np.max(np.abs(error))),
        "cell_count": int(mask.sum()),
    }


def split_metrics(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict:
    aggregate = tl_metrics(reference, prediction, mask)
    per_sample = [tl_metrics(r, p, m) for r, p, m in zip(reference, prediction, mask)]
    sample_rmse = np.asarray([item["rmse_db"] for item in per_sample])
    aggregate.update(
        {
            "macro_mean_rmse_db": float(sample_rmse.mean()),
            "median_sample_rmse_db": float(np.median(sample_rmse)),
            "p90_sample_rmse_db": float(np.percentile(sample_rmse, 90)),
            "worst_sample_rmse_db": float(sample_rmse.max()),
        }
    )
    return {"aggregate": aggregate, "per_sample": per_sample}


def high_gradient_mask(
    reference: np.ndarray, valid: np.ndarray, quantile: float = 0.9
) -> np.ndarray:
    dz = np.abs(np.diff(reference, axis=-2, prepend=reference[..., :1, :]))
    dr = np.abs(np.diff(reference, axis=-1, prepend=reference[..., :, :1]))
    magnitude = dz + dr
    threshold = np.quantile(magnitude[valid], quantile)
    return valid & (magnitude >= threshold)
