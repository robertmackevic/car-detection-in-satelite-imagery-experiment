import json
import torch

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from torch.nn import Module
from torch import Tensor


def load_config(filepath: Path) -> Optional[Dict[str, Any]]:
    with open(filepath, "r") as config:
        return json.load(config)


def count_parameters(module: Module) -> int:
    return sum(p.numel() for p in module.parameters())


def load_checkpoint(filepath: Path, model: Module) -> Module:
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["state_dict"])
    return model

def save_checkpoint(filepath: Path, model: Module) -> None:
    torch.save(model.state_dict(), filepath)


def intersection_over_union(prediction: Tensor, target: Tensor) -> Tensor:
    def get_corners(midpoint_bbox: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x1 = midpoint_bbox[..., 0:1] - midpoint_bbox[..., 2:3] / 2
        y1 = midpoint_bbox[..., 1:2] - midpoint_bbox[..., 3:4] / 2
        x2 = midpoint_bbox[..., 0:1] + midpoint_bbox[..., 2:3] / 2
        y2 = midpoint_bbox[..., 1:2] + midpoint_bbox[..., 3:4] / 2
        return x1, y1, x2, y2
    
    pred_x1, pred_y1, pred_x2, pred_y2 = get_corners(prediction)
    target_x1, target_y1, target_x2, target_y2 = get_corners(target)

    x1 = torch.max(pred_x1, target_x1)
    y1 = torch.max(pred_y1, target_y1)
    x2 = torch.min(pred_x2, target_x2)
    y2 = torch.min(pred_y2, target_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    pred_area = abs((pred_x2 - pred_x1) * (pred_y2 - pred_y1))
    target_area = abs((target_x2 - target_x1) * (target_y2 - target_y1))

    return intersection / (pred_area + target_area - intersection + 1e-6)


def non_max_suppression(bboxes: List[List[float]], iou_threshold: float, threshold: float) -> List[List[float]]:
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0] or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
            ) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms
