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
