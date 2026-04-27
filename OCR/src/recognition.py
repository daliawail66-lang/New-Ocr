"""
Text recognition module.

Recognition is the second stage of the OCR pipeline. It converts detected
text regions into machine-readable character strings.

Tesseract v4+ uses a deep learning recognizer based on a Long Short-Term
Memory (LSTM) network followed by a Connectionist Temporal Classification
(CTC) decoder. This matches the CNN-BLSTM-CTC family described in Tripathi
et al., Chapter 10 (Fig. 10.5b), which also underlies academic systems
such as CRNN, ROSETTA and STAR-Net.

IMPORTANT IMPLEMENTATION NOTE
-----------------------------
`pytesseract.image_to_data()` (used by the detection module) already runs
the full Tesseract pipeline with **full-page context** -- that is, it
performs both detection and recognition in a single call. Re-running
Tesseract on tiny word crops afterwards is both redundant and worse: the
LSTM benefits from adjacent-character context, and isolated word crops
(especially from a binarized image) lead to poor predictions.

This module therefore offers two strategies:

    * `recognize_regions()`      -- trusts the text already produced during
                                    detection. Fast and accurate.
    * `recognize_full_image()`   -- re-runs Tesseract on the full image to
                                    get a nicely formatted multi-line
                                    string (with line breaks).
    * `rerecognize_regions()`    -- forces per-region OCR with padding
                                    (kept mainly for academic comparison).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import pytesseract

from .detection import TextRegion


@dataclass
class RecognitionResult:
    """A single recognized text chunk with its source bounding box."""

    bbox: tuple                 # (x, y, w, h)
    text: str
    confidence: float
    block_num: int = 0
    line_num: int = 0
    word_num: int = 0


# ----------------------------------------------------------------------
# Primary recognition functions
# ----------------------------------------------------------------------
def recognize_full_image(
    image: np.ndarray,
    lang: str = "eng",
    psm: int = 3,
) -> str:
    """
    Run Tesseract on the entire image and return the full recognized text.

    Preserves line breaks and paragraph structure. This is the preferred
    way to obtain the final readable output, because Tesseract's LSTM
    sees the whole page at once and can leverage context across words
    and lines.

    Recommended inputs:
        * the original BGR image, or
        * a grayscale image (cv2.cvtColor(..., COLOR_BGR2GRAY))
    Avoid passing a harsh adaptive-threshold binary image here; it
    degrades the LSTM's predictions.

    Parameters
    ----------
    image : np.ndarray
        Image to recognize.
    lang : str
        Tesseract language ("eng", "fra", "ara", "eng+fra", ...).
    psm : int
        Page segmentation mode. Common values:
            3  = Fully automatic page segmentation (default)
            6  = Single uniform block of text
            7  = Single text line
            11 = Sparse text

    Returns
    -------
    str
        Multi-line recognized text, stripped of surrounding whitespace.
    """
    config = f"--psm {psm}"
    text = pytesseract.image_to_string(image, lang=lang, config=config)
    return text.strip()


def recognize_regions(
    image: np.ndarray,
    regions: List[TextRegion],
    lang: str = "eng",   # kept for API compatibility
) -> List[RecognitionResult]:
    """
    Convert detected regions into RecognitionResult objects.

    The text stored in each `TextRegion` was already produced by
    Tesseract during the detection call (`image_to_data` returns both
    bounding boxes *and* recognized text). Reusing it is both faster and
    more accurate than re-OCRing each crop, because the detection-time
    prediction was made with full-page context.

    Parameters
    ----------
    image : np.ndarray
        Unused here, kept for API symmetry and future extension.
    regions : List[TextRegion]
        Regions produced by the detection stage.
    lang : str
        Unused here; kept for signature stability.

    Returns
    -------
    List[RecognitionResult]
    """
    del image, lang  # documented no-ops

    return [
        RecognitionResult(
            bbox=r.bbox,
            text=r.text,
            confidence=r.confidence,
            block_num=r.block_num,
            line_num=r.line_num,
            word_num=r.word_num,
        )
        for r in regions
    ]


def rerecognize_regions(
    image: np.ndarray,
    regions: List[TextRegion],
    lang: str = "eng",
    padding: int = 6,
    psm: int = 8,
) -> List[RecognitionResult]:
    """
    (Optional) Force per-region OCR, with best-practice settings.

    Kept for academic comparison: some dissertations require showing that
    per-word recognition yields *worse* results than full-page recognition,
    which is exactly what happens in practice.

    Best-practice settings applied here:
        * use grayscale, not binary (LSTM prefers grayscale);
        * pad the crop by a few pixels so the LSTM sees context;
        * use --psm 8 (single word) instead of --psm 7 (single line).

    If the forced re-OCR returns empty/garbage, the detection-time text
    is used as a fallback.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    h_img, w_img = gray.shape[:2]
    config = f"--psm {psm}"
    results: List[RecognitionResult] = []

    for region in regions:
        x, y, w, h = region.bbox
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_img, x + w + padding)
        y2 = min(h_img, y + h + padding)
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        try:
            extracted = pytesseract.image_to_string(
                crop, lang=lang, config=config
            ).strip()
        except Exception:
            extracted = ""

        results.append(
            RecognitionResult(
                bbox=region.bbox,
                text=extracted if extracted else region.text,
                confidence=region.confidence,
                block_num=region.block_num,
                line_num=region.line_num,
                word_num=region.word_num,
            )
        )

    return results


# ----------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------
def results_to_text(results: List[RecognitionResult]) -> str:
    """
    Join per-region results into a clean multi-line string.

    Uses Tesseract's own (block_num, line_num) pair to group words that
    belong to the same physical line, and the word_num to order them
    within each line. This is far more reliable than y-coordinate
    bucketing and correctly reconstructs the original reading order.
    """
    if not results:
        return ""

    grouped: dict = {}
    for r in results:
        key = (r.block_num, r.line_num)
        grouped.setdefault(key, []).append(r)

    lines = []
    for key in sorted(grouped.keys()):
        words = sorted(grouped[key], key=lambda r: (r.word_num, r.bbox[0]))
        line = " ".join(w.text for w in words if w.text)
        if line:
            lines.append(line)

    return "\n".join(lines)
