from typing import Dict, Any, Tuple

import torch
from torch import Tensor
from torch.nn import Module, MSELoss

from src.utils import intersection_over_union


class YoloLoss(Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.mse = MSELoss(reduction="sum")

        self.num_x_cells, self.num_y_cells = config["grid"]
        self.num_cells = self.num_x_cells * self.num_y_cells
        self.boxes_per_cell = config["boxes_per_cell"]

        self.lambda_noobj = config["lambda_noobj"]
        self.lambda_coord = config["lambda_coord"]
        self.lambda_obj = config["lambda_obj"]

    def forward(self, predictions: Tensor, target: Tensor) -> Tuple:
        predictions = predictions.reshape(-1, self.num_y_cells, self.num_x_cells, self.boxes_per_cell * 5)

        ious = torch.cat([
            intersection_over_union(
                prediction=predictions[..., 1 + (5 * i):5 + (5 * i)],
                target=target[..., 1:5]
            ).unsqueeze(0)
            for i in range(self.boxes_per_cell)
        ], dim=0)

        best_box = torch.argmax(ious, dim=0)
        object_exists = target[..., 0].unsqueeze(3)

        # BOX LOSS
        box_predictions = self._collect_predictions(predictions, best_box, start_idx=1, end_idx=5)

        # For boxes with no object in them, set box_predictions to 0
        box_targets = object_exists * target[..., 1:5]
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_predictions = object_exists * box_predictions
        box_predictions[..., 2:4] = (
                torch.sign(box_predictions[..., 2:4]) *
                torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
        )

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # OBJECT LOSS
        pred_obj_confidence = self._collect_predictions(predictions, best_box, start_idx=0, end_idx=1)

        object_loss = self.mse(
            torch.flatten(object_exists * pred_obj_confidence),
            torch.flatten(object_exists * target[..., 0:1]),
        )

        # NO OBJECT LOSS
        no_object_loss = 0.0

        for i in range(self.boxes_per_cell):
            object_not_exists = 1 - object_exists

            no_object_loss += self.mse(
                torch.flatten(
                    input=object_not_exists * predictions[..., 5 * i:1 + (5 * i)],
                    start_dim=1
                ),
                torch.flatten(
                    input=object_not_exists * target[..., 0:1],
                    start_dim=1
                )
            )

        loss = (
                self.lambda_coord * box_loss
                + self.lambda_obj * object_loss
                + self.lambda_noobj * no_object_loss
        )

        return loss, box_loss, object_loss, no_object_loss

    def _collect_predictions(self, predictions: Tensor, best_box: Tensor, start_idx: int, end_idx: int) -> Tensor:
        collection = torch.zeros_like(predictions[..., start_idx:end_idx])

        for i in range(self.boxes_per_cell):
            box_i = predictions[..., start_idx + (5 * i):end_idx + (5 * i)]
            mask = best_box.eq(i)
            collection += mask * box_i

        return collection
