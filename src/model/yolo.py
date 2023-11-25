from typing import Dict, Any, Tuple

from torch import Tensor
from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    BatchNorm2d,
    LeakyReLU,
    MaxPool2d,
    Flatten,
    Linear,
    Dropout,
)


class Conv(Module):
    def __init__(self, in_channels: int, out_channels: int, ksize: int, stride: int, padding: int) -> None:
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, ksize, stride, padding, bias=False)
        self.batchnorm = BatchNorm2d(num_features=out_channels)
        self.leakyrelu = LeakyReLU(negative_slope=0.1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        return x


class Yolo(Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.cnn, in_channels = self._create_cnn()
        self.fc = self._create_fc(in_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.cnn(x)
        x = self.fc(x)
        return x

    def _create_cnn(self) -> Tuple[Sequential, int]:
        layers = []
        in_channels = self.config["in_channels"]
        architecture = self.config["architecture"]

        for layer in architecture:
            layer_type = layer[0]

            if layer_type == "C":
                _, ksize, out_channels, stride, padding = layer
                layers.append(Conv(in_channels, out_channels, ksize, stride, padding))
                in_channels = out_channels

            elif layer_type == "M":
                _, ksize, stride = layer
                layers.append(MaxPool2d(ksize, stride))

            elif layer_type == "B":
                _, conv1, conv2, repeat = layer
                for _ in range(repeat):
                    ksize, out_channels, stride, padding = conv1
                    layers.append(Conv(in_channels, out_channels, ksize, stride, padding))
                    in_channels = out_channels

                    ksize, out_channels, stride, padding = conv2
                    layers.append(Conv(in_channels, out_channels, ksize, stride, padding))
                    in_channels = out_channels

            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

        return Sequential(*layers), in_channels

    def _create_fc(self, in_channels: int) -> Sequential:
        num_x_cells, num_y_cells = self.config["grid"]
        num_cells = num_x_cells * num_y_cells
        boxes_per_cell = self.config["boxes_per_cell"]
        num_classes = self.config["classes"]
        dropout = self.config["dropout"]
        linear_features = 496  # Value in original paper 4096

        return Sequential(
            Flatten(),
            Linear(in_features=in_channels * num_cells, out_features=linear_features),
            Dropout(dropout),
            LeakyReLU(negative_slope=0.1),
            Linear(in_features=linear_features, out_features=num_cells * (num_classes + boxes_per_cell * 5))
        )
