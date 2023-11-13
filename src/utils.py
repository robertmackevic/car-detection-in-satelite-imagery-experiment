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


def compute_metrics(
        pred_boxes: List,
        gt_boxes: List,
        iou_threshold: float,
        num_classes: int,
) -> Dict[str, float]:
    epsilon = 1e-6
    metrics_per_class = {}

    for class_id in range(num_classes):
        detections = [box for box in pred_boxes if box[1] == class_id]
        ground_truths = [box for box in gt_boxes if box[1] == class_id]

        gt_box_per_entry_counter = dict(Counter([box[0] for box in ground_truths]))

        # Replace the count with a 0-vector
        for key, value in gt_box_per_entry_counter.items():
            gt_box_per_entry_counter[key] = torch.zeros(value)

        detections.sort(key=lambda x: x[2], reverse=True)
        tp = torch.zeros((len(detections)))
        fp = torch.zeros((len(detections)))
        fn = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            entry_idx = detection[0]
            entry_gt_boxes = [bbox for bbox in ground_truths if bbox[0] == entry_idx]

            best_iou, best_gt_idx = 0, 0

            for gt_idx, gt in enumerate(entry_gt_boxes):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou > iou_threshold:
                if gt_box_per_entry_counter[entry_idx][best_gt_idx] == 0:
                    tp[detection_idx] = 1
                    gt_box_per_entry_counter[entry_idx][best_gt_idx] = 1
                else:
                    fp[detection_idx] = 1
            else:
                fn[detection_idx] = 1

        tp_cum_sum = torch.cumsum(tp, dim=0)
        fp_cum_sum = torch.cumsum(fp, dim=0)
        fn_cum_sum = torch.cumsum(fn, dim=0)

        recalls = torch.divide(tp_cum_sum, (tp_cum_sum + fn_cum_sum + epsilon))
        recalls = torch.cat((torch.tensor([0]), recalls))

        precisions = torch.divide(tp_cum_sum, (tp_cum_sum + fp_cum_sum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))

        ap = torch.trapz(precisions, recalls)

        # Calculate F1 score
        f1 = 2 * (precisions * recalls) / (precisions + recalls + epsilon)

        # Store metrics for the class
        metrics_per_class[class_id] = {
            "AP": ap.item(),
            "Precision": precisions[-1].item(),
            "Recall": recalls[-1].item(),
            "F1": f1[-1].item(),
            "TP": tp_cum_sum[-1].item(),
            "FP": fp_cum_sum[-1].item(),
            "FN": fn_cum_sum[-1].item(),
        }

    aggregated_metrics = {
        "mAP": _aggregate_mean(metrics_per_class, metric="AP"),
        "Precision": _aggregate_mean(metrics_per_class, metric="Precision"),
        "Recall": _aggregate_mean(metrics_per_class, metric="Recall"),
        "F1": _aggregate_mean(metrics_per_class, metric="F1"),
        "TP": _aggregate_sum(metrics_per_class, metric="TP"),
        "FP": _aggregate_sum(metrics_per_class, metric="FP"),
        "FN": _aggregate_sum(metrics_per_class, metric="FN"),
    }

    return aggregated_metrics


def _aggregate_sum(metrics_per_class: Dict[int, Dict[str, Any]], metric: str) -> float:
    return sum(metrics_per_class[class_id][metric] for class_id in metrics_per_class.keys())


def _aggregate_mean(metrics_per_class: Dict[int, Dict[str, Any]], metric: str) -> float:
    return _aggregate_sum(metrics_per_class, metric) / len(metrics_per_class.keys())
