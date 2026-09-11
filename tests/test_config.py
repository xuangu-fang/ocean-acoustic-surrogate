from pathlib import Path

import pytest

from ocean_acoustic_surrogate.config import MVPConfig

ROOT = Path(__file__).resolve().parents[1]


def test_mvp_contract_is_frozen():
    config = MVPConfig.from_yaml(ROOT / "configs/mvp.yaml")
    assert config.contract.frequency_hz == 1000.0
    assert config.contract.source_depth_m == 50.0
    assert config.contract.range_end_m == 50000.0
    assert config.contract.water_depth_m == 2000.0
    assert config.contract.field_mode == "incoherent"
    assert len(config.config_hash) == 64


def test_realistic_contract_has_balanced_real_data_anchors():
    config = MVPConfig.from_yaml(ROOT / "configs/realistic_mvp.yaml")
    assert config.contract.frequency_hz == 1000.0
    assert config.contract.reference_num_rays == 25600
    assert config.ssp_family.name == "bashi_woa23_june_narrow"
    assert len(config.contract.bathymetry.profiles) == 4
    assert (
        min(depth for profile in config.contract.bathymetry.profiles for depth in profile.depths_m)
        == 2000.0
    )


def test_terrain_tradeoff_contract_keeps_one_complexity_axis():
    config = MVPConfig.from_yaml(ROOT / "configs/realistic_terrain_mvp.yaml")
    assert config.ssp_family.name == "bashi_woa23_june_narrow"
    assert not config.ssp_family.profiles
    assert len(config.contract.bathymetry.profiles) == 4
    relief = [
        max(profile.depths_m) - min(profile.depths_m)
        for profile in config.contract.bathymetry.profiles
    ]
    assert min(relief) >= 1900.0
    assert max(max(profile.depths_m) for profile in config.contract.bathymetry.profiles) <= 4800.0
    assert max(len(profile.depths_m) for profile in config.contract.bathymetry.profiles) <= 12


def test_multi_source_depth_contract_is_explicit_and_bounded():
    config = MVPConfig.from_yaml(ROOT / "configs/multi_source_depth_mvp.yaml")
    assert config.contract.resolved_source_depths_m == (
        50.0,
        100.0,
        200.0,
        400.0,
        700.0,
        1000.0,
    )
    assert max(config.contract.resolved_source_depths_m) < 2000.0


def test_source_may_be_deeper_than_receiver_grid_when_local_seabed_is_deeper():
    config = MVPConfig.from_yaml(ROOT / "configs/multi_source_depth_mvp.yaml")
    payload = config.model_dump(mode="json")
    payload["contract"]["source_depths_m"] = [50.0, 2000.0]
    payload["contract"]["bathymetry"]["profiles"] = payload["contract"]["bathymetry"]["profiles"][
        :2
    ]
    deep = MVPConfig.model_validate(payload)
    assert max(deep.contract.resolved_source_depths_m) > deep.contract.depth_end_m

    payload["contract"]["source_depths_m"] = [4800.0]
    with pytest.raises(ValueError, match="source-position seabed"):
        MVPConfig.model_validate(payload)
