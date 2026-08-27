import numpy as np

from ocean_acoustic_surrogate.config import MVPConfig
from ocean_acoustic_surrogate.dataset import _dataset_splits


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
