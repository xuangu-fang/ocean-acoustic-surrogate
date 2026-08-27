import numpy as np
import torch

from ocean_acoustic_surrogate.features import TargetTransform, build_features


def test_feature_channels_are_explicit():
    profiles = np.full((2, 4), 1500.0, dtype=np.float32)
    ranges = np.linspace(100, 50000, 8, dtype=np.float32)
    assert build_features(profiles, ranges, use_hankel=False).shape == (2, 1, 4, 8)
    assert build_features(profiles, ranges, use_hankel=True).shape == (2, 2, 4, 8)
    bathymetry = np.linspace(2000, 2080, 8, dtype=np.float32)
    assert (
        build_features(
            profiles,
            ranges,
            use_hankel=False,
            bathymetry_depths_m=bathymetry,
        ).shape
        == (2, 2, 4, 8)
    )


def test_target_transform_round_trip():
    targets = np.arange(48, dtype=np.float32).reshape(3, 4, 4)
    mask = np.ones_like(targets, dtype=bool)
    transform = TargetTransform.fit(targets[:2], mask[:2])
    encoded = torch.from_numpy(transform.encode(targets))[:, None]
    decoded = transform.decode_tensor(encoded).numpy()
    assert np.allclose(decoded, targets)
