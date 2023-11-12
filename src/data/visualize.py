from typing import Tuple, List

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from src.data.entry import ImageEntry

DEFAULT_FIGSIZE = (12, 6)
DEFAULT_LINEWIDTH = 2
COLOR_GREEN = (0, 255, 0)


def annotate_entry_image(
        entry: ImageEntry,
        color: Tuple[int, int, int] = COLOR_GREEN,
        linewidth: int = DEFAULT_LINEWIDTH
) -> np.ndarray:
    image = entry.image.copy()
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    for anno in entry.annotations:
        half_width = anno.bbox.width / 2
        half_height = anno.bbox.height / 2

        x1 = int((anno.bbox.x + half_width) * entry.image.shape[1])
        y1 = int((anno.bbox.y + half_height) * entry.image.shape[0])
        x2 = int((anno.bbox.x - half_width) * entry.image.shape[1])
        y2 = int((anno.bbox.y - half_height) * entry.image.shape[0])

        image = cv.rectangle(image, (x1, y1), (x2, y2), color, linewidth)

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return image


def plot_single_entry_original_and_annotated(
        entry: ImageEntry,
        figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
        color: Tuple[int, int, int] = COLOR_GREEN,
        linewidth: int = DEFAULT_LINEWIDTH,
) -> None:
    annotated_image = annotate_entry_image(entry, color, linewidth)

    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(entry.image)

    plt.subplot(1, 2, 2)
    plt.title("Annotated")
    plt.imshow(annotated_image)

    plt.show()


def plot_entries_original_and_annotated(
        entries: List[ImageEntry],
        samples_to_display: int,
        figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
        color: Tuple[int, int, int] = COLOR_GREEN,
        linewidth: int = DEFAULT_LINEWIDTH,
) -> None:
    for entry in entries[:samples_to_display]:
        plot_single_entry_original_and_annotated(entry, figsize, color, linewidth)
