from dataclasses import replace
from typing import Dict, Any, List

import torch
from PIL import Image
from torch import Tensor
from torch.nn import Module

from src.data.entry import ImageEntry, entry_to_patches, Annotation, patches_to_entry
from src.data.transforms import get_inference_transform
from src.utils import non_max_suppression


class Detector:
    def __init__(self, config: Dict[str, Any], model: Module):
        self.model = model
        self.num_x_cells, self.num_y_cells = config["grid"]
        self.num_cells = self.num_x_cells * self.num_y_cells

        self.boxes_per_cell = config["boxes_per_cell"]
        self.image_size = config["image_size"]
        self.iou_threshold = config["iou_threshold"]

        self.transform = get_inference_transform(
            image_size=config["image_size"],
            in_channels=config["in_channels"],
        )

    def detect(self, entry: ImageEntry, threshold: float = 0.5) -> ImageEntry:
        patches = entry_to_patches(entry, patch_size=self.image_size)
        source = torch.cat([self.transform(Image.fromarray(patch.image)).unsqueeze(0) for patch in patches], dim=0)
        batch_size = source.shape[0]

        self.model.eval()
        with torch.no_grad():
            output = self.model(source)

        predicted_bboxes = self.cellboxes_to_boxes(output)
        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                predicted_bboxes[idx],
                iou_threshold=self.iou_threshold,
                threshold=threshold,
            )
            annotations = []
            for bbox in nms_boxes:
                annotations.append(Annotation(
                    x=bbox[1],
                    y=bbox[2],
                    width=bbox[3],
                    height=bbox[4],
                ))

            patches[idx] = replace(patches[idx], annotations=annotations)

        result = patches_to_entry(patches)
        return result

    def cellboxes_to_boxes(self, grid: Tensor) -> List:
        converted_pred = self.convert_cellboxes(grid).reshape(grid.shape[0], self.num_cells, -1)
        converted_pred[..., 0] = converted_pred[..., 0].long()
        all_bboxes = []

        for batch_idx in range(grid.shape[0]):
            bboxes = []

            for bbox_idx in range(self.num_cells):
                bboxes.append([x.item() for x in converted_pred[batch_idx, bbox_idx, :]])
            all_bboxes.append(bboxes)

        return all_bboxes

    def convert_cellboxes(self, grid: Tensor) -> Tensor:
        grid = grid.to("cpu")
        batch_size = grid.shape[0]

        grid = grid.reshape(
            batch_size, self.num_y_cells, self.num_x_cells, self.boxes_per_cell * 5)

        bboxes = [
            grid[..., 1 + (5 * i): 5 + (5 * i)]
            for i in range(self.boxes_per_cell)
        ]

        scores = torch.cat([
            grid[..., 5 * i].unsqueeze(0)
            for i in range(self.boxes_per_cell)
        ], dim=0)

        best_box = scores.argmax(0).unsqueeze(-1)
        best_boxes = torch.zeros_like(bboxes[0])

        for i in range(self.boxes_per_cell):
            best_boxes += best_box.eq(i).float() * bboxes[i]

        cell_indices = torch.arange(self.num_y_cells).repeat(batch_size, self.num_x_cells, 1).unsqueeze(-1)

        x = 1 / self.num_x_cells * (best_boxes[..., :1] + cell_indices)
        y = 1 / self.num_y_cells * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
        w_y = 1 / self.num_y_cells * best_boxes[..., 2:4]

        converted_bboxes = torch.cat((x, y, w_y), dim=-1)

        best_confidence = torch.max(
            torch.stack([
                grid[..., 5 * i] for i in range(self.boxes_per_cell)
            ], dim=0), dim=0).values.unsqueeze(-1)

        return torch.cat((best_confidence, converted_bboxes), dim=-1)
