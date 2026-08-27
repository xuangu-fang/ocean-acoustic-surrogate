"""Compact anisotropic FNO variants for the fixed range-depth grid."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, channels: int, modes_depth: int, modes_range: int) -> None:
        super().__init__()
        self.channels = channels
        self.modes_depth = modes_depth
        self.modes_range = modes_range
        scale = 1.0 / channels
        shape = (channels, channels, modes_depth, modes_range)
        self.weight_top = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weight_bottom = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    @staticmethod
    def _multiply(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", values, weights)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft2(values, norm="ortho")
        modes_depth = min(self.modes_depth, values.shape[-2] // 2)
        modes_range = min(self.modes_range, spectrum.shape[-1])
        output = spectrum.new_zeros(
            values.shape[0], self.channels, values.shape[-2], spectrum.shape[-1]
        )
        output[:, :, :modes_depth, :modes_range] = self._multiply(
            spectrum[:, :, :modes_depth, :modes_range],
            self.weight_top[:, :, :modes_depth, :modes_range],
        )
        output[:, :, -modes_depth:, :modes_range] = self._multiply(
            spectrum[:, :, -modes_depth:, :modes_range],
            self.weight_bottom[:, :, :modes_depth, :modes_range],
        )
        return torch.fft.irfft2(output, s=values.shape[-2:], norm="ortho")


class FNOBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        modes_depth: int,
        modes_range: int,
        *,
        residual: bool,
        local_kernel_size: int,
    ) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(channels, modes_depth, modes_range)
        if local_kernel_size == 1:
            self.local = nn.Conv2d(channels, channels, 1)
        else:
            padding = local_kernel_size // 2
            self.local = nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    local_kernel_size,
                    padding=padding,
                    groups=channels,
                ),
                nn.GELU(),
                nn.Conv2d(channels, channels, 1),
            )
        self.norm = nn.GroupNorm(1, channels)
        self.residual = residual

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.norm(self.spectral(values) + self.local(values))
        if self.residual:
            update = update + values
        return F.gelu(update)


class FNO2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        modes_depth: int,
        modes_range: int,
        n_layers: int,
        padding_depth: int = 0,
        padding_range: int = 0,
        residual_blocks: bool = False,
        local_kernel_size: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.padding_depth = padding_depth
        self.padding_range = padding_range
        self.lift = nn.Sequential(
            nn.Conv2d(in_channels + 2, hidden_channels, 1),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            FNOBlock(
                hidden_channels,
                modes_depth,
                modes_range,
                residual=residual_blocks,
                local_kernel_size=local_kernel_size,
            )
            for _ in range(n_layers)
        )
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, 1, 1),
        )

    @staticmethod
    def coordinates(values: torch.Tensor) -> torch.Tensor:
        batch, _, depth, distance = values.shape
        z = torch.linspace(0, 1, depth, dtype=values.dtype, device=values.device)
        r = torch.linspace(0, 1, distance, dtype=values.dtype, device=values.device)
        z = z.view(1, 1, depth, 1).expand(batch, 1, depth, distance)
        r = r.view(1, 1, 1, distance).expand(batch, 1, depth, distance)
        return torch.cat((z, r), dim=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or values.shape[1] != self.in_channels:
            raise ValueError(f"expected [batch,{self.in_channels},depth,range]")
        values = torch.cat((values, self.coordinates(values)), dim=1)
        values = self.lift(values)
        if self.padding_depth or self.padding_range:
            values = F.pad(
                values,
                (0, self.padding_range, 0, self.padding_depth),
                mode="replicate",
            )
        for block in self.blocks:
            values = block(values)
        if self.padding_depth:
            values = values[..., : -self.padding_depth, :]
        if self.padding_range:
            values = values[..., : -self.padding_range]
        return self.project(values)


def build_model(config: dict, in_channels: int) -> FNO2d:
    return FNO2d(
        in_channels=in_channels,
        hidden_channels=int(config["hidden_channels"]),
        modes_depth=int(config["modes_depth"]),
        modes_range=int(config["modes_range"]),
        n_layers=int(config["n_layers"]),
        padding_depth=int(config.get("padding_depth", 0)),
        padding_range=int(config.get("padding_range", 0)),
        residual_blocks=bool(config.get("residual_blocks", False)),
        local_kernel_size=int(config.get("local_kernel_size", 1)),
    )
