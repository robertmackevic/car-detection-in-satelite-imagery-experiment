import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2 as cv
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from src.paths import CONFIG_FILE


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cv.setRNGSeed(seed)


def load_config() -> Dict[str, Any]:
    with open(CONFIG_FILE, "r") as config:
        return json.load(config)


def save_config(config: Dict[str, Any], filepath: Path) -> None:
    with open(filepath, "w") as file:
        json.dump(config, file)


def count_parameters(module: Module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_layers(module: Module) -> int:
    return len(list(module.named_modules()))


def load_checkpoint(filepath: Path, model: Module) -> Module:
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint)
    return model


def save_checkpoint(filepath: Path, model: Module) -> None:
    torch.save(model.state_dict(), filepath)


def _get_corners(midpoint_bbox: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    x1 = midpoint_bbox[..., 0:1] - midpoint_bbox[..., 2:3] / 2
    y1 = midpoint_bbox[..., 1:2] - midpoint_bbox[..., 3:4] / 2
    x2 = midpoint_bbox[..., 0:1] + midpoint_bbox[..., 2:3] / 2
    y2 = midpoint_bbox[..., 1:2] + midpoint_bbox[..., 3:4] / 2
    return x1, y1, x2, y2


def intersection_over_union(prediction: Tensor, target: Tensor) -> Tensor:
    pred_x1, pred_y1, pred_x2, pred_y2 = _get_corners(prediction)
    target_x1, target_y1, target_x2, target_y2 = _get_corners(target)

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
    bboxes = [box for box in bboxes if box[0] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[0], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box for box in bboxes
            if intersection_over_union(
                torch.tensor(chosen_box[1:]),
                torch.tensor(box[1:]),
            ) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def compute_metrics(
        pred_boxes: List,
        gt_boxes: List,
        iou_threshold: float,
) -> Dict[str, float]:
    epsilon = 1e-6

    detections = [box for box in pred_boxes]
    ground_truths = [box for box in gt_boxes]

    gt_box_per_entry_counter = dict(Counter([box[0] for box in ground_truths]))

    # Replace the count with a 0-vector
    for key, value in gt_box_per_entry_counter.items():
        gt_box_per_entry_counter[key] = torch.zeros(value)

    detections.sort(key=lambda x: x[1], reverse=True)
    tps = torch.zeros((len(detections)))
    fps = torch.zeros((len(detections)))
    ious = torch.zeros((len(detections)))
    total_true_bboxes = len(ground_truths)

    for detection_idx, detection in enumerate(detections):
        entry_idx = detection[0]
        entry_gt_boxes = [bbox for bbox in ground_truths if bbox[0] == entry_idx]

        best_iou, best_gt_idx = 0, 0

        for gt_idx, gt in enumerate(entry_gt_boxes):
            iou = intersection_over_union(
                torch.tensor(detection[2:]),
                torch.tensor(gt[2:]),
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        ious[detection_idx] = best_iou

        if best_iou > iou_threshold:
            if gt_box_per_entry_counter[entry_idx][best_gt_idx] == 0:
                tps[detection_idx] = 1
                gt_box_per_entry_counter[entry_idx][best_gt_idx] = 1
            else:
                fps[detection_idx] = 1
        else:
            fps[detection_idx] = 1

    tp_cum_sum = torch.cumsum(tps, dim=0) if tps.numel() != 0 else Tensor([0, ])
    fp_cum_sum = torch.cumsum(fps, dim=0) if fps.numel() != 0 else Tensor([0, ])

    recalls = torch.divide(tp_cum_sum, (total_true_bboxes + epsilon))
    recalls = torch.cat((torch.tensor([0]), recalls))

    precisions = torch.divide(tp_cum_sum, (tp_cum_sum + fp_cum_sum + epsilon))
    precisions = torch.cat((torch.tensor([1]), precisions))

    ap = torch.trapz(precisions, recalls)
    f1 = 2 * (precisions * recalls) / (precisions + recalls + epsilon)

    metrics = {
        "AP": ap.item(),
        "IoU": ious.mean().item(),
        "Precision": precisions[-1].item(),
        "Recall": recalls[-1].item(),
        "F1": f1[-1].item(),
        "TP": tp_cum_sum[-1].item(),
        "FP": fp_cum_sum[-1].item(),
    }

    return metrics
