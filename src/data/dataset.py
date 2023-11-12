from typing import List, Tuple, Dict, Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Grayscale, Resize, ToTensor

from src.data.entry import ImageEntry, split_entries_positive_negative


class ObjectDetectionDataset(Dataset):
    def __init__(self, entries: List[ImageEntry], single_class: bool, config: Dict[str, Any]) -> None:
        self.entries = entries
        self.num_classes = self._get_number_of_classes()
        self.num_positive, self.num_negative = self._get_number_of_positive_negative_entries()
        self.single_class = single_class

        self.num_splits = config["splits"]
        self.num_boxes = config["boxes"]
        self.num_classes = config["classes"]

        image_size = config["image_size"]

        self.transform = Compose([
            Grayscale(num_output_channels=1),
            Resize((image_size, image_size)),
            ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        image = Image.fromarray(self.entries[index].image)
        image = self.transform(image)

        cells = torch.zeros((self.num_splits, self.num_splits, self.num_classes + 5 * self.num_boxes))

        for annotation in self.entries[index].annotations:
            split_x = self.num_splits * annotation.bbox.x
            split_y = self.num_splits * annotation.bbox.y

            cell_row = int(split_y)
            cell_column = int(split_x)

            cell_x = split_x - cell_column
            cell_y = split_y - cell_row

            cell_w = annotation.bbox.width * self.num_splits
            cell_h = annotation.bbox.height * self.num_splits

            # One object per cell
            if cells[cell_row, cell_column, self.num_classes] == 0:
                cells[cell_row, cell_column, self.num_classes] = 1

                bbox = Tensor([cell_x, cell_y, cell_w, cell_h])
                cells[cell_row, cell_column, self.num_classes + 1: self.num_classes + 5] = bbox

                class_id = 0 if self.single_class else annotation.class_id
                # One-hot encoding class_id
                cells[cell_row, cell_column, class_id] = 1

        return image, cells

    def _get_number_of_classes(self) -> int:
        return len(set(
            annotation.class_id
            for entry in self.entries
            for annotation in entry.annotations
        ))

    def _get_number_of_positive_negative_entries(self) -> Tuple[int, int]:
        positive, negative = split_entries_positive_negative(self.entries)
        return len(positive), len(negative)

    def describe(self, name: str) -> str:
        objects_per_entry = [len(entry.annotations) for entry in self.entries]
        total_objects = sum(objects_per_entry)

        description = f"Number {name} of entries: " \
                      + f"{len(self.entries)} | " \
                      + f"positive {self.num_positive} | " \
                      + f"negative {self.num_negative} | " \
                      + f"objects {total_objects} | " \
                      + f"max objects {max(objects_per_entry)} | " \
                      + f"avg objects {total_objects / self.num_positive:.2f}"

        return description
