"""
Text detection module.

Text detection is the first stage of the OCR pipeline. Its goal is to
*locate* regions of an image that contain text, producing bounding boxes
around each word or line. It does NOT try to read the characters.

This implementation uses Tesseract's internal layout analyzer, exposed
through `pytesseract.image_to_data()`. Tesseract performs connected-component
analysis on the binary image: it groups neighboring dark pixels into
candidate characters, then clusters them into words and text lines based
on geometry. Each candidate is scored with a confidence value.

Typology (see Chapter 10 of Tripathi et al., Smart Text Reader System):
    Detection methods = Classical ML (sliding window, connected components)
                      U Deep learning (bounding-box regression, segmentation,
                        hybrid)
    Tesseract falls in the "connected components" branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


@dataclass
class TextRegion:
    """A single detected text region (word-level)."""

    x: int
    y: int
    w: int
    h: int
    text: str
    confidence: float
    block_num: int = 0
    line_num: int = 0
    word_num: int = 0

    @property
    def bbox(self) -> tuple:
        """Return the bounding box as (x, y, w, h)."""
        return (self.x, self.y, self.w, self.h)


def detect_text_regions(
    image: np.ndarray,
    confidence_threshold: float = 30.0,
    lang: str = "eng",
) -> List[TextRegion]:
    """
    Detect all text regions in an image.

    Internally calls Tesseract's full analysis pipeline, which performs
    layout analysis AND character recognition with full-page context.
    The returned `TextRegion` objects therefore contain *already-recognized*
    text in their `text` attribute, which can be used directly without any
    further OCR call.

    The block/line/word numbers returned by Tesseract are also preserved;
    they are the most reliable way to reconstruct the original reading
    order.

    Parameters
    ----------
    image : np.ndarray
        Grayscale or BGR image. Tesseract handles both.
    confidence_threshold : float
        Minimum Tesseract confidence (0-100) to keep a region.
    lang : str
        Tesseract language model (e.g. "eng", "fra", "ara").

    Returns
    -------
    List[TextRegion]
        One entry per detected word-level region.
    """
    data = pytesseract.image_to_data(
        image, lang=lang, output_type=Output.DICT
    )

    regions: List[TextRegion] = []
    n = len(data["text"])

    for i in range(n):
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            continue

        text = data["text"][i].strip()
        if not text or conf < confidence_threshold:
            continue

        regions.append(
            TextRegion(
                x=int(data["left"][i]),
                y=int(data["top"][i]),
                w=int(data["width"][i]),
                h=int(data["height"][i]),
                text=text,
                confidence=conf,
                block_num=int(data.get("block_num", [0] * n)[i]),
                line_num=int(data.get("line_num", [0] * n)[i]),
                word_num=int(data.get("word_num", [0] * n)[i]),
            )
        )

    return regions


def draw_bounding_boxes(
    image: np.ndarray,
    regions: List[TextRegion],
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    show_confidence: bool = False,
) -> np.ndarray:
    """
    Draw bounding boxes around the detected regions (for visualization).

    Parameters
    ----------
    image : np.ndarray
        Original image (BGR or grayscale). Grayscale is converted to BGR.
    regions : List[TextRegion]
        Regions returned by `detect_text_regions`.
    color : tuple
        Box color in BGR. Default: green.
    thickness : int
        Line thickness in pixels.
    show_confidence : bool
        If True, overlay the confidence value above each box.
    """
    output = image.copy()
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    for region in regions:
        x, y, w, h = region.bbox
        cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)

        if show_confidence:
            label = f"{region.confidence:.0f}%"
            cv2.putText(
                output,
                label,
                (x, max(y - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    return output


def detection_stats(regions: List[TextRegion]) -> dict:
    """Compute simple statistics over the detected regions."""
    if not regions:
        return {
            "count": 0,
            "mean_confidence": 0.0,
            "min_confidence": 0.0,
            "max_confidence": 0.0,
        }

    confs = [r.confidence for r in regions]
    return {
        "count": len(regions),
        "mean_confidence": float(np.mean(confs)),
        "min_confidence": float(np.min(confs)),
        "max_confidence": float(np.max(confs)),
    }
