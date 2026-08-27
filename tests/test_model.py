import pytest
import torch

from ocean_acoustic_surrogate.models import FNO2d


@pytest.mark.parametrize("padding,local", [((0, 0), 1), ((4, 8), 3)])
def test_fno_preserves_grid_shape(padding, local):
    model = FNO2d(
        in_channels=2,
        hidden_channels=8,
        modes_depth=4,
        modes_range=6,
        n_layers=2,
        padding_depth=padding[0],
        padding_range=padding[1],
        residual_blocks=True,
        local_kernel_size=local,
    )
    values = torch.randn(3, 2, 16, 32)
    output = model(values)
    assert output.shape == (3, 1, 16, 32)
    output.mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_zero_output_initialization_starts_from_residual_anchor():
    model = FNO2d(
        in_channels=2,
        hidden_channels=8,
        modes_depth=4,
        modes_range=6,
        n_layers=2,
        zero_output_init=True,
    )
    output = model(torch.randn(3, 2, 16, 32))
    assert torch.count_nonzero(output) == 0
