from typing import Dict, Any, Tuple

import torch
from torch import Tensor
from torch.nn import Module, MSELoss

from src.utils import intersection_over_union


class YoloLoss(Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.mse = MSELoss(reduction="sum")

        self.num_splits = config["splits"]
        self.num_boxes = config["boxes"]
        self.num_classes = config["classes"]

        # Significance weights for no-object and bbox coords
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions: Tensor, target: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        predictions = predictions.reshape(-1, self.num_splits, self.num_splits, self.num_classes + self.num_boxes * 5)
        bbox_idx_start = self.num_classes + 1
        bbox_idx_end = self.num_classes + 5
        score_idx_start = self.num_classes
        score_idx_end = bbox_idx_start

        ious = torch.cat([
            intersection_over_union(
                prediction=predictions[..., bbox_idx_start + (5 * i):bbox_idx_end + (5 * i)],
                target=target[..., bbox_idx_start:bbox_idx_end]
            ).unsqueeze(0)
            for i in range(self.num_boxes)
        ], dim=0)

        bestbox = torch.argmax(ious, dim=0)
        object_exists = target[..., self.num_classes].unsqueeze(3)

        # BOX LOSS
        box_predictions = torch.zeros_like(predictions[..., bbox_idx_start:bbox_idx_end])

        for i in range(self.num_boxes):
            box_i = predictions[..., bbox_idx_start + (5 * i):bbox_idx_end + (5 * i)]
            mask = bestbox.eq(i)
            box_predictions += mask * box_i

        # For boxes with no object in them, set box_predictions to 0
        box_predictions = object_exists * box_predictions

        box_targets = object_exists * target[..., bbox_idx_start:bbox_idx_end]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # OBJECT LOSS
        pred_obj_confidence = torch.zeros_like(predictions[..., score_idx_start:score_idx_end])

        for i in range(self.num_boxes):
            box_i = predictions[..., score_idx_start + (5 * i):score_idx_end + (5 * i)]
            mask = bestbox.eq(i)
            pred_obj_confidence += mask * box_i

        object_loss = self.mse(
            torch.flatten(object_exists * pred_obj_confidence),
            torch.flatten(object_exists * target[..., score_idx_start:score_idx_end]),
        )

        # NO OBJECT LOSS
        no_object_loss = 0.0

        for i in range(self.num_boxes):
            object_not_exists = 1 - object_exists

            no_object_loss += self.mse(
                torch.flatten(
                    input=object_not_exists * predictions[..., score_idx_start + (5 * i):score_idx_end + (5 * i)],
                    start_dim=1
                ),
                torch.flatten(
                    input=object_not_exists * target[..., score_idx_start:score_idx_end],
                    start_dim=1
                )
            )

        # CLASS LOSS
        class_loss = self.mse(
            torch.flatten(object_exists * predictions[..., :self.num_classes], end_dim=-2, ),
            torch.flatten(object_exists * target[..., :self.num_classes], end_dim=-2, ),
        )

        loss = (
                self.lambda_coord * box_loss
                + object_loss
                + self.lambda_noobj * no_object_loss
                + class_loss
        )

        return loss, box_loss, object_loss, no_object_loss, class_loss
