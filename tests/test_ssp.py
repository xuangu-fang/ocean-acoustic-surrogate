import numpy as np

from ocean_acoustic_surrogate.config import MVPConfig
from ocean_acoustic_surrogate.ssp import assign_splits, build_ssp_records


def test_ssp_design_is_reproducible(tmp_path):
    config = MVPConfig.from_yaml("configs/mvp.yaml")
    first = build_ssp_records(config.ssp_family, 16, 12)
    second = build_ssp_records(config.ssp_family, 16, 12)
    assert np.allclose(first[3].speeds_mps, second[3].speeds_mps)
    assert min(record.speeds_mps.min() for record in first) >= 1475.0
    assert max(record.speeds_mps.max() for record in first) <= 1565.0


def test_split_counts_cover_all_samples():
    splits = assign_splits(512, seed=1, train=0.75, validation=0.125)
    assert (splits == "train").sum() == 384
    assert (splits == "validation").sum() == 64
    assert (splits == "test").sum() == 64


def test_nested_design_preserves_frozen_128_sample_prefix():
    config = MVPConfig.from_yaml("configs/realistic_terrain_mvp.yaml")
    original = build_ssp_records(config.ssp_family, 128, config.contract.seed)
    extended = build_ssp_records(config.ssp_family, 256, config.contract.seed)
    assert [record.sample_id for record in original] == [
        record.sample_id for record in extended[:128]
    ]
    assert all(
        np.array_equal(left.parameters, right.parameters)
        and np.array_equal(left.speeds_mps, right.speeds_mps)
        for left, right in zip(original, extended[:128])
    )
