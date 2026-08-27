import numpy as np

from ocean_acoustic_surrogate.field_validation import _reference_features


def test_reference_features_encode_ssp_bathymetry_and_task_parameters():
    metadata = {
        "reference_task": {
            "frequency_hz": 500.0,
            "source_depth_m": 50.0,
            "environment": {
                "water_depth_m": 2000.0,
                "bathymetry": {
                    "points": [
                        {"range_m": 0.0, "depth_m": 1800.0},
                        {"range_m": 1000.0, "depth_m": 2000.0},
                    ]
                },
                "ssp": {
                    "points": [
                        {"depth_m": 0.0, "speed_mps": 1500.0},
                        {"depth_m": 1000.0, "speed_mps": 1450.0},
                    ],
                    "grid": None,
                },
            },
        }
    }
    ranges = np.asarray([0.0, 1000.0], dtype=np.float32)
    depths = np.asarray([0.0, 1000.0], dtype=np.float32)
    mask = np.asarray([[True, True], [True, False]])

    features = _reference_features(metadata, ranges, depths, mask)

    assert features.shape == (5, 2, 2)
    np.testing.assert_allclose(features[0, :, 0], [0.0, -1.0])
    np.testing.assert_allclose(features[1, 0], [0.9, 1.0])
    np.testing.assert_allclose(features[2], 0.5)
    np.testing.assert_allclose(features[3], 0.025)
    np.testing.assert_array_equal(features[4], mask.astype(np.float32))
