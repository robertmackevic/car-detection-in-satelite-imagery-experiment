from typing import Dict, Any

import torch
from torch import Tensor
from torch.nn import (
    Module,
    Conv2d,
    BatchNorm2d,
    LeakyReLU,
    Sequential,
    ModuleList,
    Upsample,
    Sigmoid
)


class ConvBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super(ConvBlock, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batch_norm = BatchNorm2d(out_channels)
        self.activation = LeakyReLU(negative_slope=0.1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x


class ResidualBlock(Module):
    def __init__(self, in_channels: int, use_residual: bool, num_repeats: int = 1) -> None:
        super(ResidualBlock, self).__init__()
        self.use_residual = use_residual
        self.num_repeats = num_repeats

        hidden = in_channels // 2
        self.layers = ModuleList([
            Sequential(
                ConvBlock(in_channels, out_channels=hidden, stride=1, kernel_size=1, padding=0),
                ConvBlock(hidden, out_channels=in_channels, stride=1, kernel_size=3, padding=1)
            ) for _ in range(num_repeats)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = x + layer(x) if self.use_residual else layer(x)
        return x


class ScaleBlock(Module):

    def __init__(self, in_channels: int) -> None:
        super(ScaleBlock, self).__init__()
        self.res = ResidualBlock(in_channels, use_residual=False, num_repeats=1)
        self.conv = ConvBlock(in_channels, out_channels=in_channels // 2, stride=1, kernel_size=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.res(x)
        x = self.conv(x)
        return x


class PredictionBlock(Module):
    def __init__(self, in_channels: int) -> None:
        super(PredictionBlock, self).__init__()
        hidden = in_channels * 2

        self.prediction = Sequential(
            ConvBlock(in_channels, out_channels=hidden, stride=1, kernel_size=3, padding=1),
            Conv2d(in_channels=hidden, out_channels=3, stride=1, kernel_size=1, padding=0),
            Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.prediction(x).permute(0, 3, 2, 1)


class Yolo(Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super(Yolo, self).__init__()
        self.in_channels = config["in_channels"]
        self.architecture = config["architecture"]
        self.network = self._create_network()

    def _create_network(self) -> ModuleList:
        layers = ModuleList()
        in_channels = self.in_channels

        for layer in self.architecture:
            layer_type = layer[0]

            if layer_type == "C":
                _, kernel_size, out_channels, stride, padding = layer
                layers.append(ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ))
                in_channels = out_channels

            elif layer_type == "R":
                _, num_repeats = layer
                layers.append(ResidualBlock(
                    in_channels=in_channels,
                    use_residual=True,
                    num_repeats=num_repeats,
                ))

            elif layer_type == "U":
                _, scale_factor = layer
                layers.append(Upsample(
                    scale_factor=scale_factor,
                ))
                in_channels = in_channels * 3

            elif layer_type == "S":
                layers.append(ScaleBlock(
                    in_channels=in_channels,
                ))
                in_channels = in_channels // 2

            elif layer_type == "P":
                layers += [
                    ScaleBlock(in_channels),
                    PredictionBlock(in_channels=in_channels // 2)
                ]

        return layers

    def forward(self, x: Tensor) -> Tensor:
        route_connections = []

        for layer in self.network:
            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return x
