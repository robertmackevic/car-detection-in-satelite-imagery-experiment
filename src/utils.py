import json
from collections import Counter
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import torch
from torch import Tensor
from torch.nn import Module


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


def mean_average_precision(
        pred_boxes: List,
        true_boxes: List,
        iou_threshold: float,
        num_classes: int,
) -> float:
    epsilon = 1e-6
    aps = []

    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        tp = torch.zeros((len(detections)))
        fp = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    tp[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    fp[detection_idx] = 1

            else:
                fp[detection_idx] = 1

        TP_cumsum = torch.cumsum(tp, dim=0)
        FP_cumsum = torch.cumsum(fp, dim=0)

        recalls = TP_cumsum / (total_true_bboxes + epsilon)

        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))

        recalls = torch.cat((torch.tensor([0]), recalls))

        aps.append(torch.trapz(precisions, recalls))

    return sum(aps) / len(aps)
