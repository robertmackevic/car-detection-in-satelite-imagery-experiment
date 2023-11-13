from typing import Dict, Any

import torch
from torch import Tensor
from torch.nn import Module


class Detector:
    def __init__(self, config: Dict[str, Any], model: Module):
        self.model = model
        self.num_splits = config["splits"]
        self.num_cells = self.num_splits ** 2
        self.num_boxes = config["boxes"]
        self.num_classes = config["classes"]

    def detect(self, source: Tensor):
        self.model.eval()

        with torch.no_grad():
            _, output = self.model(source)

        predicted_bboxes = self.cellboxes_to_boxes(output)
        return predicted_bboxes

    def cellboxes_to_boxes(self, cells: Tensor):
        converted_pred = self.convert_cellboxes(cells).reshape(cells.shape[0], self.num_cells, -1)
        converted_pred[..., 0] = converted_pred[..., 0].long()
        all_bboxes = []

        for ex_idx in range(cells.shape[0]):
            bboxes = []

            for bbox_idx in range(self.num_cells):
                bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
            all_bboxes.append(bboxes)

        return all_bboxes

    def convert_cellboxes(self, cells: Tensor) -> Tensor:
        cells = cells.to("cpu")
        batch_size = cells.shape[0]

        cells = cells.reshape(batch_size, self.num_splits, self.num_splits, self.num_classes + self.num_boxes * 5)
        bboxes = [cells[..., self.num_classes + 1 + (5 * i): self.num_classes + 5 + (5 * i)] for i in
                  range(self.num_boxes)]

        scores = torch.cat([
            cells[..., self.num_classes + (5 * i)].unsqueeze(0)
            for i in range(self.num_boxes)
        ], dim=0)

        best_box = scores.argmax(0).unsqueeze(-1)
        best_boxes = torch.zeros_like(bboxes[0])
        for i in range(self.num_boxes):
            best_boxes += best_box.eq(i).float() * bboxes[i]

        cell_indices = torch.arange(self.num_splits).repeat(batch_size, self.num_splits, 1).unsqueeze(-1)

        x = 1 / self.num_splits * (best_boxes[..., :1] + cell_indices)
        y = 1 / self.num_splits * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
        w_y = 1 / self.num_splits * best_boxes[..., 2:4]

        converted_bboxes = torch.cat((x, y, w_y), dim=-1)

        predicted_class = cells[..., :self.num_classes].argmax(-1).unsqueeze(-1)

        best_confidence = torch.max(
            torch.stack([
                cells[..., self.num_classes + (5 * i)] for i in range(self.num_boxes)
            ], dim=0), dim=0).values.unsqueeze(-1)

        converted_preds = torch.cat((predicted_class, best_confidence, converted_bboxes), dim=-1)
        return converted_preds
