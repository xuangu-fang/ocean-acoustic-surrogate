import numpy as np

from ocean_acoustic_surrogate.metrics import split_metrics, stratified_metrics, tl_metrics


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


def test_stratified_metrics_reports_each_sample_group():
    reference = np.zeros((4, 2, 2), dtype=np.float32)
    prediction = np.stack([np.ones((2, 2)) * value for value in (1, 1, 3, 3)])
    groups = np.asarray(["winter", "winter", "summer", "summer"])
    metrics = stratified_metrics(
        reference, prediction, np.ones_like(reference, dtype=bool), groups
    )
    assert metrics["winter"]["rmse_db"] == 1.0
    assert metrics["summer"]["rmse_db"] == 3.0
