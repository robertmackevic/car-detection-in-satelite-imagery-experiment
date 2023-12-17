from typing import List, Tuple, Dict, Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from src.data.entry import ImageEntry


class ObjectDetectionDataset(Dataset):
    def __init__(self, entries: List[ImageEntry], config: Dict[str, Any], transforms: Compose) -> None:
        self.entries = entries
        self.num_negative = sum(entry.is_negative for entry in entries)
        self.num_positive = len(entries) - self.num_negative
        self.num_x_cells, self.num_y_cells = config["grid"]
        self.transform = transforms

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        image = Image.fromarray(self.entries[index].image)
        image = self.transform(image)

        grid = torch.zeros((self.num_y_cells, self.num_x_cells, 3))

        for annotation in self.entries[index].annotations:
            x = self.num_x_cells * annotation.x
            y = self.num_y_cells * annotation.y

            cell_row = int(y)
            cell_column = int(x)

            # One object per cell
            if grid[cell_row, cell_column, 0] == 0:
                grid[cell_row, cell_column, 0] = 1

                center = Tensor([x - cell_column, y - cell_row])
                grid[cell_row, cell_column, 1:3] = center

        return image, grid

    def describe(self, name: str) -> str:
        objects_per_entry = [entry.num_objects for entry in self.entries]
        total_objects = sum(objects_per_entry)

        description = f"Number of {name} entries: " \
                      + f"{len(self.entries)} | " \
                      + f"positive {self.num_positive} | " \
                      + f"negative {self.num_negative} | " \
                      + f"objects {total_objects} | " \
                      + f"max objects {max(objects_per_entry)} | " \
                      + f"avg objects {total_objects / self.num_positive:.2f}"

        return description
