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
