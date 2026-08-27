"""Configuration and storage contracts for the narrow MVP."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class SeabedConfig(BaseModel):
    kind: str = "fluid"
    speed_mps: float = 1700.0
    density_kgm3: float = 2000.0
    attenuation_dbperlambda: float = 0.8


class BathymetryProfileConfig(BaseModel):
    name: str
    ranges_m: list[float]
    depths_m: list[float]
    source: str
    processing: str

    @model_validator(mode="after")
    def validate_profile(self) -> BathymetryProfileConfig:
        if len(self.ranges_m) != len(self.depths_m) or len(self.ranges_m) < 2:
            raise ValueError("bathymetry ranges/depths must have equal length >= 2")
        if self.ranges_m != sorted(self.ranges_m):
            raise ValueError("bathymetry ranges must be ascending")
        if min(self.depths_m) <= 0:
            raise ValueError("bathymetry depths must be positive")
        return self


class BathymetryFamilyConfig(BaseModel):
    name: str
    profiles: list[BathymetryProfileConfig]

    @model_validator(mode="after")
    def validate_family(self) -> BathymetryFamilyConfig:
        if not self.profiles:
            raise ValueError("bathymetry family requires at least one profile")
        names = [profile.name for profile in self.profiles]
        if len(names) != len(set(names)):
            raise ValueError("bathymetry profile names must be unique")
        return self


class ContractConfig(BaseModel):
    dataset_id: str
    seed: int
    frequency_hz: float = 1000.0
    source_depth_m: float = 50.0
    water_depth_m: float = 2000.0
    range_start_m: float = 100.0
    range_end_m: float = 50000.0
    range_steps: int = 256
    depth_start_m: float = 10.0
    depth_end_m: float = 1990.0
    depth_steps: int = 96
    field_mode: str = "incoherent"
    reference_num_rays: int = 25600
    pilot_ray_counts: list[int] = Field(default_factory=lambda: [3200, 6400, 12800, 25600])
    invalid_tl_fill_db: float = 120.0
    seabed: SeabedConfig = Field(default_factory=SeabedConfig)
    bathymetry: BathymetryFamilyConfig | None = None

    @model_validator(mode="after")
    def validate_fixed_acceptance_domain(self) -> ContractConfig:
        if self.frequency_hz != 1000.0:
            raise ValueError("MVP contract requires frequency_hz=1000")
        if self.range_end_m != 50000.0:
            raise ValueError("MVP contract requires range_end_m=50000")
        minimum_bottom = (
            min(depth for profile in self.bathymetry.profiles for depth in profile.depths_m)
            if self.bathymetry is not None
            else self.water_depth_m
        )
        if self.depth_end_m >= minimum_bottom:
            raise ValueError("depth_end_m must remain above the seabed")
        if self.bathymetry is not None:
            for profile in self.bathymetry.profiles:
                if profile.ranges_m[0] > 0:
                    raise ValueError("bathymetry must start at or before source range 0")
                if profile.ranges_m[-1] < self.range_end_m:
                    raise ValueError("bathymetry must cover the full calculation range")
        if self.field_mode != "incoherent":
            raise ValueError("MVP label contract requires incoherent Bellhop TL")
        return self


class SSPFamilyConfig(BaseModel):
    name: str
    depths_m: list[float]
    base_speeds_mps: list[float]
    global_offset_mps: tuple[float, float]
    thermocline_amplitude_mps: tuple[float, float]
    channel_axis_shift_m: tuple[float, float]
    deep_gradient_mps: tuple[float, float]
    interpolation_points: int = 33

    @model_validator(mode="after")
    def validate_profile(self) -> SSPFamilyConfig:
        if len(self.depths_m) != len(self.base_speeds_mps):
            raise ValueError("depths_m and base_speeds_mps must have equal length")
        if self.depths_m != sorted(self.depths_m):
            raise ValueError("SSP depths must be ascending")
        return self


class SplitConfig(BaseModel):
    train_fraction: float = 0.75
    validation_fraction: float = 0.125
    test_fraction: float = 0.125

    @model_validator(mode="after")
    def validate_sum(self) -> SplitConfig:
        if abs(self.train_fraction + self.validation_fraction + self.test_fraction - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to one")
        return self


class AcceptanceConfig(BaseModel):
    maximum_test_rmse_db: float = 2.0
    maximum_test_mae_db: float = 2.0
    maximum_p95_latency_ms: float = 100.0


class StorageConfig(BaseModel):
    default_root: Path


class MVPConfig(BaseModel):
    contract: ContractConfig
    ssp_family: SSPFamilyConfig
    split: SplitConfig
    acceptance: AcceptanceConfig
    storage: StorageConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> MVPConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

    def canonical_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def artifact_root(self) -> Path:
        return Path(os.environ.get("OCEAN_SURROGATE_ROOT", self.storage.default_root))

    @property
    def dataset_root(self) -> Path:
        return self.artifact_root / "datasets" / self.contract.dataset_id


def load_campaign(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    if not data.get("experiments"):
        raise ValueError("campaign requires at least one experiment")
    return data
