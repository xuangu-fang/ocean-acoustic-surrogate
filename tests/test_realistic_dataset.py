import numpy as np

from ocean_acoustic_surrogate.config import MVPConfig
from ocean_acoustic_surrogate.dataset import (
    _dataset_splits,
    _source_depth_for_record,
    build_task,
)
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


def test_multi_source_depth_design_has_72_balanced_groups_and_reusable_50m_cases():
    config = MVPConfig.from_yaml("configs/multi_source_depth_mvp.yaml")
    previous = MVPConfig.from_yaml("configs/realistic_seasonal_terrain_mvp.yaml")
    splits = _dataset_splits(config, 432)
    records = build_ssp_records(
        config.ssp_family,
        432,
        config.contract.seed,
        template_cycle_stride=4,
    )
    previous_records = build_ssp_records(
        previous.ssp_family,
        384,
        previous.contract.seed,
        template_cycle_stride=4,
    )
    groups = [
        (index % 4, record.profile_name, _source_depth_for_record(config, record))
        for index, record in enumerate(records)
    ]
    assert len(set(groups)) == 72
    for group in set(groups):
        indices = np.asarray([index for index, value in enumerate(groups) if value == group])
        assert len(indices) == 6
        assert dict(zip(*np.unique(splits[indices], return_counts=True))) == {
            "test": 1,
            "train": 4,
            "validation": 1,
        }

    reusable = [
        index
        for index, record in enumerate(records[:384])
        if _source_depth_for_record(config, record) == 50.0
    ]
    assert len(reusable) == 72
    assert all(
        records[index].profile_name == previous_records[index].profile_name
        and np.array_equal(records[index].parameters, previous_records[index].parameters)
        and np.array_equal(records[index].speeds_mps, previous_records[index].speeds_mps)
        for index in reusable
    )
    task = build_task(config, records[12], 12800, "source_depth_contract")
    assert task.source_depth_m == 100.0
    assert task.metadata["source_depth_m"] == 100.0
