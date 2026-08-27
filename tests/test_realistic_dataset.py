import numpy as np

from ocean_acoustic_surrogate.config import MVPConfig
from ocean_acoustic_surrogate.dataset import _dataset_splits
from ocean_acoustic_surrogate.ssp import build_ssp_records


def test_realistic_splits_are_balanced_by_terrain_template():
    config = MVPConfig.from_yaml("configs/realistic_mvp.yaml")
    splits = _dataset_splits(config, 128)
    for profile_index in range(4):
        local = splits[np.arange(profile_index, 128, 4)]
        assert dict(zip(*np.unique(local, return_counts=True))) == {
            "test": 4,
            "train": 24,
            "validation": 4,
        }


def test_seasonal_splits_cover_every_terrain_month_pair():
    config = MVPConfig.from_yaml("configs/realistic_seasonal_mvp.yaml")
    splits = _dataset_splits(config, 128)
    for group_index in range(4):
        local = splits[np.arange(group_index, 128, 4)]
        assert dict(zip(*np.unique(local, return_counts=True))) == {
            "test": 4,
            "train": 24,
            "validation": 4,
        }
    records = build_ssp_records(config.ssp_family, 8, config.contract.seed)
    assert [record.profile_name for record in records] == [
        "woa23_january",
        "woa23_april",
        "woa23_july",
        "woa23_october",
    ] * 2
