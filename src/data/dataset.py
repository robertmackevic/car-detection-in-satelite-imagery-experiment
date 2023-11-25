from typing import List, Tuple, Dict, Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from src.data.entry import ImageEntry
from src.data.transforms import get_inference_transform


class ObjectDetectionDataset(Dataset):
    def __init__(self, entries: List[ImageEntry], config: Dict[str, Any]) -> None:
        self.entries = entries
        self.num_negative = sum(entry.is_negative for entry in entries)
        self.num_positive = len(entries) - self.num_negative

        self.num_x_cells, self.num_y_cells = config["grid"]
        self.boxes_per_cell = config["boxes_per_cell"]
        self.num_classes = config["classes"]

        self.transform = get_inference_transform(
            image_size=config["image_size"],
            in_channels=config["in_channels"],
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        image = Image.fromarray(self.entries[index].image)
        image = self.transform(image)

        grid = torch.zeros((self.num_y_cells, self.num_x_cells, self.num_classes + 5 * self.boxes_per_cell))

        for annotation in self.entries[index].annotations:
            x = self.num_x_cells * annotation.x
            y = self.num_y_cells * annotation.y

            cell_row = int(y)
            cell_column = int(x)

            cell_x = x - cell_column
            cell_y = y - cell_row

            cell_w = annotation.width * self.num_x_cells
            cell_h = annotation.height * self.num_y_cells

            # One object per cell
            if grid[cell_row, cell_column, self.num_classes] == 0:
                grid[cell_row, cell_column, self.num_classes] = 1

                bbox = Tensor([cell_x, cell_y, cell_w, cell_h])
                grid[cell_row, cell_column, self.num_classes + 1: self.num_classes + 5] = bbox

                class_id = 0 if self.num_classes == 1 else annotation.class_id
                # One-hot encoding class_id
                grid[cell_row, cell_column, class_id] = 1

        return image, grid

    def describe(self, name: str) -> str:
        objects_per_entry = [entry.num_objects for entry in self.entries]
        total_objects = sum(objects_per_entry)

        description = f"Number {name} of entries: " \
                      + f"{len(self.entries)} | " \
                      + f"positive {self.num_positive} | " \
                      + f"negative {self.num_negative} | " \
                      + f"objects {total_objects} | " \
                      + f"max objects {max(objects_per_entry)} | " \
                      + f"avg objects {total_objects / self.num_positive:.2f}"

        return description
