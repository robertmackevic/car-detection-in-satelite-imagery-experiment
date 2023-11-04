import random
import numpy as np
import cv2 as cv

from dataclasses import dataclass
from typing import Tuple, List
from pathlib import Path
from os import listdir

IMAGE_FORMAT_SUFFIX = ".jpg"
ANNOTATION_FORMAT_SUFFIX = ".txt"


@dataclass(frozen=True)
class YoloBbox:
    x: float
    y: float
    width: float
    height: float

@dataclass(frozen=True)
class Annotation:
    class_id: int
    bbox: YoloBbox

    @classmethod
    def from_yolo_annotation(cls, line: str) -> "Annotation":
        parts = line.strip().split()
        class_id = int(parts[0])
        bbox = YoloBbox(
            x=float(parts[1]),
            y=float(parts[2]),
            width=float(parts[3]),
            height=float(parts[4]),
        )

        return cls(class_id, bbox)


@dataclass(frozen=True)
class ImageEntry:
    image: np.ndarray
    annotations: List[Annotation]

    @classmethod
    def from_image_filepath(cls, image_filepath: Path) -> "ImageEntry":
        """
        Creates an ImageEntry object from a given filepath which leads to the image.
        It is assumed that:
            * Both image and the corresponding annotations files are in the same directory.
            * Both image and the corresponding annotations files share the same name apart from the suffix.
            * Annotations are saved in YOLO format.
        """
        annotations: List[Annotation] = []
        annotations_filepath = image_filepath.with_suffix(ANNOTATION_FORMAT_SUFFIX)
              
        if annotations_filepath.is_file():
            with open(annotations_filepath, "r") as file:
                for line in file:
                    annotations.append(Annotation.from_yolo_annotation(line))
        
        bgr_image = cv.imread(str(image_filepath))
        rgb_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
        image = np.asarray(rgb_image)

        return cls(image, annotations)


def read_entries_from_directory(data_dir: Path) -> List[ImageEntry]:
    entries: List[ImageEntry] = []

    for filename in listdir(data_dir):
        filepath = data_dir / filename

        if filepath.is_dir():
            entries_from_other_dir = read_entries_from_directory(filepath)
            entries += entries_from_other_dir
            continue

        if filepath.suffix == IMAGE_FORMAT_SUFFIX:
            entries.append(ImageEntry.from_image_filepath(filepath))
    
    return entries

def split_entries_train_val_test(
        entries: List[ImageEntry],
        val_fraction: float = .1,
        test_fraction: float = .1,
        seed: int = None,
    )-> Tuple[List[ImageEntry], List[ImageEntry], List[ImageEntry]]:
    
    if test_fraction + val_fraction >= 1.0:
        raise ValueError("Error: test_fraction + val_fraction >= 1.0")

    positive_entries, negative_entries = split_entries_positive_negative(entries)
    
    num_val_pos_entries = int(len(positive_entries) * val_fraction)
    num_val_neg_entries = int(len(negative_entries) * val_fraction)

    num_test_pos_entries = int(len(positive_entries) * test_fraction)
    num_test_neg_entries = int(len(negative_entries) * test_fraction)

    random.seed(seed)
    random.shuffle(positive_entries)
    random.shuffle(negative_entries)

    val_set = positive_entries[:num_val_pos_entries] + negative_entries[:num_val_neg_entries]
    random.shuffle(val_set)

    test_set = positive_entries[num_val_pos_entries:num_val_pos_entries + num_test_pos_entries] + \
        negative_entries[num_val_neg_entries:num_val_neg_entries + num_test_neg_entries]
    random.shuffle(test_set)

    train_set = positive_entries[num_val_pos_entries + num_test_pos_entries:] + \
        negative_entries[num_val_neg_entries + num_test_neg_entries:]
    random.shuffle(train_set)

    return train_set, val_set, test_set


def split_entries_positive_negative(entries: List[ImageEntry]) -> Tuple[List[ImageEntry], List[ImageEntry]]:
    positive, negative = [], []
    for entry in entries:
        if entry.annotations:
            positive.append(entry)
        else:
            negative.append(entry)

    return positive, negative
