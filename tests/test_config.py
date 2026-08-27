from pathlib import Path

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
    assert min(
        depth
        for profile in config.contract.bathymetry.profiles
        for depth in profile.depths_m
    ) == 2000.0


def test_terrain_tradeoff_contract_keeps_one_complexity_axis():
    config = MVPConfig.from_yaml(ROOT / "configs/realistic_terrain_mvp.yaml")
    assert config.ssp_family.name == "bashi_woa23_june_narrow"
    assert not config.ssp_family.profiles
    assert len(config.contract.bathymetry.profiles) == 4
    relief = [max(profile.depths_m) - min(profile.depths_m) for profile in config.contract.bathymetry.profiles]
    assert min(relief) >= 400.0
    assert max(len(profile.depths_m) for profile in config.contract.bathymetry.profiles) == 6
