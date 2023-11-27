from typing import Dict, Any, Tuple

import torch
from torch import Tensor
from torch.nn import Module, MSELoss


class YoloLoss(Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.mse = MSELoss(reduction="sum")

        self.num_x_cells, self.num_y_cells = config["grid"]
        self.num_cells = self.num_x_cells * self.num_y_cells

        self.lambda_noobj = config["lambda_noobj"]
        self.lambda_coord = config["lambda_coord"]
        self.lambda_obj = config["lambda_obj"]

    def forward(self, predictions: Tensor, target: Tensor) -> Tuple:
        predictions = predictions.reshape(-1, self.num_y_cells, self.num_x_cells, 5)
        object_exists = target[..., 0].unsqueeze(3)

        # BOX LOSS
        box_targets = object_exists * target[..., 1:5]
        box_predictions = object_exists * predictions[..., 1:5]

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        box_predictions[..., 2:4] = (
                torch.sign(box_predictions[..., 2:4]) *
                torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
        )

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # OBJECT LOSS
        object_loss = self.mse(
            torch.flatten(object_exists * predictions[..., 0:1], start_dim=1),
            torch.flatten(object_exists * target[..., 0:1], start_dim=1),
        )

        # NO OBJECT LOSS
        object_not_exists = 1 - object_exists
        no_object_loss = self.mse(
            torch.flatten(object_not_exists * predictions[..., 0:1], start_dim=1),
            torch.flatten(object_not_exists * target[..., 0:1], start_dim=1),
        )

        loss = (
                self.lambda_coord * box_loss
                + self.lambda_obj * object_loss
                + self.lambda_noobj * no_object_loss
        )

        return loss, box_loss, object_loss, no_object_loss
