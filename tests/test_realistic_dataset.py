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


def test_extended_realistic_splits_are_balanced_by_terrain_template():
    config = MVPConfig.from_yaml("configs/realistic_terrain_mvp.yaml")
    splits = _dataset_splits(config, 256)
    for profile_index in range(4):
        local = splits[np.arange(profile_index, 256, 4)]
        assert dict(zip(*np.unique(local, return_counts=True))) == {
            "test": 8,
            "train": 48,
            "validation": 8,
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


def test_three_month_terrain_design_is_balanced_and_prefix_stable():
    config = MVPConfig.from_yaml("configs/realistic_seasonal_terrain_mvp.yaml")
    splits = _dataset_splits(config, 384)
    for group_index in range(12):
        local = splits[np.arange(group_index, 384, 12)]
        assert dict(zip(*np.unique(local, return_counts=True))) == {
            "test": 4,
            "train": 24,
            "validation": 4,
        }

    pilot = build_ssp_records(
        config.ssp_family,
        96,
        config.contract.seed,
        template_cycle_stride=4,
    )
    full = build_ssp_records(
        config.ssp_family,
        384,
        config.contract.seed,
        template_cycle_stride=4,
    )
    assert all(
        left.profile_name == right.profile_name
        and np.array_equal(left.parameters, right.parameters)
        and np.array_equal(left.speeds_mps, right.speeds_mps)
        for left, right in zip(pilot, full[:96])
    )
    groups = [(record.profile_name, index % 4) for index, record in enumerate(full)]
    assert all(groups.count(group) == 32 for group in set(groups))
