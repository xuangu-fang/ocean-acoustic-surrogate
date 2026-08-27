import numpy as np

from ocean_acoustic_surrogate.metrics import split_metrics, tl_metrics


def test_tl_metrics_are_in_db():
    reference = np.zeros((2, 3), dtype=np.float32)
    prediction = np.ones((2, 3), dtype=np.float32) * 2
    metrics = tl_metrics(reference, prediction, np.ones_like(reference, dtype=bool))
    assert metrics["rmse_db"] == 2.0
    assert metrics["mae_db"] == 2.0


def test_split_metrics_reports_macro_and_worst():
    reference = np.zeros((2, 2, 2), dtype=np.float32)
    prediction = np.stack((np.ones((2, 2)), np.ones((2, 2)) * 3))
    metrics = split_metrics(reference, prediction, np.ones_like(reference, dtype=bool))
    assert metrics["aggregate"]["macro_mean_rmse_db"] == 2.0
    assert metrics["aggregate"]["worst_sample_rmse_db"] == 3.0
