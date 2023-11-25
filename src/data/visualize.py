from typing import Tuple, List, Optional

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from src.data.entry import ImageEntry

DEFAULT_FIGSIZE1 = (6, 6)
DEFAULT_FIGSIZE2 = (12, 6)
DEFAULT_LINEWIDTH = 2
COLOR_GREEN = (0, 255, 0)
COLOR_BLACK = (0, 0, 0)


def annotate_entry_image(
        entry: ImageEntry,
        annotation_style: str = "bbox",
        color: Tuple[int, int, int] = COLOR_GREEN,
        linewidth: int = DEFAULT_LINEWIDTH
) -> np.ndarray:
    image = entry.image.copy()
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    if annotation_style not in ("bbox", "circle"):
        raise ValueError("Available annotation styles: bbox, circle.")

    for anno in entry.annotations:
        half_width = anno.width / 2
        half_height = anno.height / 2

        if annotation_style == "bbox":
            x1 = int((anno.x + half_width) * entry.image.shape[1])
            y1 = int((anno.y + half_height) * entry.image.shape[0])
            x2 = int((anno.x - half_width) * entry.image.shape[1])
            y2 = int((anno.y - half_height) * entry.image.shape[0])
            cv.rectangle(image, (x1, y1), (x2, y2), color, linewidth)

        elif annotation_style == "circle":
            w, h = entry.image.shape[0], entry.image.shape[1]
            radius = anno.height * h * .25 if h < w else anno.width * w * .25
            x = int(anno.x * w)
            y = int(anno.y * h)
            cv.circle(image, (x, y), int(radius), COLOR_BLACK, cv.FILLED)
            cv.circle(image, (x, y), int(radius * .75), color, cv.FILLED)

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return image


def plot_single_entry_original_and_annotated(
        entry: ImageEntry,
        annotation_style: str = "bbox",
        figsize: Tuple[int, int] = DEFAULT_FIGSIZE2,
        color: Tuple[int, int, int] = COLOR_GREEN,
        linewidth: int = DEFAULT_LINEWIDTH,
) -> None:
    annotated_image = annotate_entry_image(entry, annotation_style, color, linewidth)

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
        samples_to_display: Optional[int] = None,
        annotation_style: str = "bbox",
        figsize: Tuple[int, int] = DEFAULT_FIGSIZE2,
        color: Tuple[int, int, int] = COLOR_GREEN,
        linewidth: int = DEFAULT_LINEWIDTH,
) -> None:
    samples_to_display = len(entries) if samples_to_display is None else samples_to_display
    for entry in entries[:samples_to_display]:
        plot_single_entry_original_and_annotated(entry, annotation_style, figsize, color, linewidth)


def plot_entry(
        entry: ImageEntry,
        annotate: bool,
        annotation_style: str = "bbox",
        figsize: Tuple[int, int] = DEFAULT_FIGSIZE1,
        color: Tuple[int, int, int] = COLOR_GREEN,
        linewidth: int = DEFAULT_LINEWIDTH,
) -> None:
    image = annotate_entry_image(entry, annotation_style, color, linewidth) if annotate else entry.image
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.show()
