import random
from dataclasses import dataclass
from os import listdir
from pathlib import Path
from typing import Tuple, List, Optional

import cv2 as cv
import numpy as np

IMAGE_FORMAT_SUFFIX = ".jpg"
ANNOTATION_FORMAT_SUFFIX = ".txt"


@dataclass(frozen=True)
class Annotation:
    x: float
    y: float
    width: float
    height: float

    @classmethod
    def from_yolo_string(cls, line: str) -> "Annotation":
        parts = line.strip().split()
        return cls(
            x=float(parts[1]),
            y=float(parts[2]),
            width=float(parts[3]),
            height=float(parts[4]),
        )


@dataclass(frozen=True)
class ImageEntry:
    image: np.ndarray
    annotations: List[Annotation]
    is_negative: bool
    num_objects: int

    @classmethod
    def from_image_filepath(cls, image_filepath: Path) -> "ImageEntry":
        """
        Creates an ImageEntry object from a given filepath which leads to the image.
        It is assumed that:
            * Both image and the corresponding annotation files are in the same directory.
            * Both image and the corresponding annotation files share the same name apart from the suffix.
            * Annotations are saved in YOLO format.
        """
        annotations: List[Annotation] = []
        annotations_filepath = image_filepath.with_suffix(ANNOTATION_FORMAT_SUFFIX)

        if annotations_filepath.is_file():
            with open(annotations_filepath, "r") as file:
                for line in file:
                    annotations.append(Annotation.from_yolo_string(line))

        return cls(
            image=np.asarray(cv.cvtColor(cv.imread(str(image_filepath)), cv.COLOR_BGR2RGB)),
            annotations=annotations,
            is_negative=(len(annotations) == 0),
            num_objects=len(annotations),
        )


def read_entries_from_directory(
        data_dir: Path,
        leave_out_negatives: bool = False,
        patch_size: Optional[int] = None,
        max_entries: Optional[int] = None
) -> List[ImageEntry]:
    entries: List[ImageEntry] = []

    for filename in listdir(data_dir):
        if max_entries is not None and len(entries) >= max_entries:
            break

        filepath = data_dir / filename

        if filepath.is_dir():
            entries_from_other_dir = read_entries_from_directory(
                filepath, leave_out_negatives, patch_size, max_entries)

            entries += entries_from_other_dir
            continue

        if filepath.suffix == IMAGE_FORMAT_SUFFIX:
            entry = ImageEntry.from_image_filepath(filepath)

            if leave_out_negatives and entry.is_negative:
                continue

            if patch_size is not None:
                patches = entry_to_patches(entry, patch_size)
                entries += patches

            else:
                entries.append(entry)

    return entries


def entry_to_patches(entry: ImageEntry, patch_size: int) -> List[ImageEntry]:
    height, width = entry.image.shape[:2]

    if height != width:
        raise ValueError("Image should be 1:1 resolution")

    if width % patch_size != 0:
        raise ValueError("Image dimensions should be divisible by the patch size")

    image_size_in_patches = width // patch_size
    patches = []

    for row in range(image_size_in_patches):
        row_patch_start_idx = row * patch_size
        row_patch_end_idx = (row + 1) * patch_size

        for col in range(image_size_in_patches):
            col_patch_start_idx = col * patch_size
            col_patch_end_idx = (col + 1) * patch_size

            patch_image = entry.image[row_patch_start_idx:row_patch_end_idx, col_patch_start_idx:col_patch_end_idx, :]
            patch_annotations = []

            for annotation in entry.annotations:
                if (  # Check if the annotation bbox is within the current patch
                        col_patch_start_idx / width <= annotation.x < col_patch_end_idx / width and
                        row_patch_start_idx / height <= annotation.y < row_patch_end_idx / height
                ):
                    # Adjust the coordinates relative to the patch
                    bbox_x = image_size_in_patches * annotation.x
                    bbox_y = image_size_in_patches * annotation.y

                    bbox_row_idx = int(bbox_y)
                    bbox_col_idx = int(bbox_x)

                    relative_annotation = Annotation(
                        x=bbox_x - bbox_col_idx,
                        y=bbox_y - bbox_row_idx,
                        width=annotation.width * image_size_in_patches,
                        height=annotation.height * image_size_in_patches,
                    )

                    patch_annotations.append(relative_annotation)

            patches.append(ImageEntry(
                image=patch_image,
                annotations=patch_annotations,
                is_negative=(len(patch_annotations) == 0),
                num_objects=len(patch_annotations),
            ))

    return patches


def patches_to_entry(patches: List[ImageEntry]) -> ImageEntry:
    patch_size = patches[0].image.shape[0]
    image_size_in_patches = int(np.sqrt(len(patches)))
    image_size = image_size_in_patches * patch_size
    num_channels = patches[0].image.shape[2]
    merged_image = np.zeros((image_size, image_size, num_channels), dtype=np.uint8)
    merged_annotations = []

    for patch_idx, patch in enumerate(patches):
        row_idx = patch_idx // image_size_in_patches
        col_idx = patch_idx % image_size_in_patches

        row_patch_start_idx = row_idx * patch_size
        row_patch_end_idx = (row_idx + 1) * patch_size
        col_patch_start_idx = col_idx * patch_size
        col_patch_end_idx = (col_idx + 1) * patch_size

        merged_image[row_patch_start_idx:row_patch_end_idx, col_patch_start_idx:col_patch_end_idx, :] = patch.image

        for annotation in patch.annotations:
            bbox_x = col_idx * patch_size + annotation.x * patch_size
            bbox_y = row_idx * patch_size + annotation.y * patch_size

            image_annotation = Annotation(
                x=bbox_x / image_size,
                y=bbox_y / image_size,
                width=annotation.width * patch_size / image_size,
                height=annotation.height * patch_size / image_size,
            )

            merged_annotations.append(image_annotation)

    return ImageEntry(
        image=merged_image,
        annotations=merged_annotations,
        is_negative=(len(merged_annotations) == 0),
        num_objects=len(merged_annotations),
    )


def equalize_negative_samples_with_positives(entries: List[ImageEntry]) -> List[ImageEntry]:
    positive, negative = split_entries_positive_negative(entries)
    if len(negative) > len(positive):
        equalized = positive + negative[:len(positive)]
        random.shuffle(equalized)
        return equalized
    return entries


def split_entries_train_val_test(
        entries: List[ImageEntry],
        val_fraction: float = .1,
        test_fraction: float = .1
) -> Tuple[List[ImageEntry], List[ImageEntry], List[ImageEntry]]:
    if test_fraction + val_fraction >= 1.0:
        raise ValueError("Error: test_fraction + val_fraction >= 1.0")

    positive_entries, negative_entries = split_entries_positive_negative(entries)

    num_val_pos_entries = int(len(positive_entries) * val_fraction)
    num_val_neg_entries = int(len(negative_entries) * val_fraction)

    num_test_pos_entries = int(len(positive_entries) * test_fraction)
    num_test_neg_entries = int(len(negative_entries) * test_fraction)

    random.shuffle(positive_entries)
    random.shuffle(negative_entries)

    val_set = positive_entries[:num_val_pos_entries] + negative_entries[:num_val_neg_entries]
    random.shuffle(val_set)

    test_set = (
            positive_entries[num_val_pos_entries:num_val_pos_entries + num_test_pos_entries] +
            negative_entries[num_val_neg_entries:num_val_neg_entries + num_test_neg_entries])
    random.shuffle(test_set)

    train_set = (
            positive_entries[num_val_pos_entries + num_test_pos_entries:] +
            negative_entries[num_val_neg_entries + num_test_neg_entries:])
    random.shuffle(train_set)

    return train_set, val_set, test_set


def split_entries_positive_negative(entries: List[ImageEntry]) -> Tuple[List[ImageEntry], List[ImageEntry]]:
    positive, negative = [], []
    for entry in entries:
        if entry.is_negative:
            negative.append(entry)
        else:
            positive.append(entry)

    return positive, negative
