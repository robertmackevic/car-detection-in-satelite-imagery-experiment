from typing import Dict, Any, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss


class YoloLoss(Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super(YoloLoss, self).__init__()

        self.reduction = "sum"
        self.lambda_coord = config["lambda_coord"]
        self.lambda_obj = config["lambda_obj"]
        self.lambda_no_obj = config["lambda_no_obj"]

    def forward(self, prediction: Tensor, target: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        obj = target[..., 0] == 1
        no_obj = torch.logical_not(obj)

        num_obj_cells = torch.sum(obj)
        num_empty_cells = torch.sum(no_obj)
        total_cells = num_obj_cells + num_empty_cells

        obj_weight = total_cells / (2 * num_obj_cells)
        no_obj_weight = total_cells / (2 * num_empty_cells)

        object_loss = binary_cross_entropy_with_logits(
            torch.flatten(prediction[..., 0:1][obj]),
            torch.flatten(target[..., 0:1][obj]),
            weight=obj_weight,
            reduction=self.reduction
        )

        no_object_loss = binary_cross_entropy_with_logits(
            torch.flatten(prediction[..., 0:1][no_obj]),
            torch.flatten(target[..., 0:1][no_obj]),
            weight=no_obj_weight,
            reduction=self.reduction
        )

        coord_loss = mse_loss(
            prediction[..., 1:3][obj],
            target[..., 1:3][obj],
            reduction=self.reduction
        )

        loss = (
                self.lambda_coord * coord_loss +
                self.lambda_obj * object_loss +
                self.lambda_no_obj * no_object_loss
        )

        return loss, coord_loss, object_loss, no_object_loss
