"""
Generate a sample test image with known text, so the app can be tested
without needing to download anything.
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont


OUTPUT_PATH = os.path.join("assets", "sample.png")
LINES = [
    "Smart Text Reader",
    "",
    "Optical Character Recognition",
    "for visually impaired users.",
    "",
    "Chapter 1 : Detection and Recognition",
    "Python + OpenCV + Tesseract",
]


def _get_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\calibri.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def generate_sample(output_path: str = OUTPUT_PATH) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    width, height = 900, 500
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    title_font = _get_font(36)
    body_font = _get_font(24)

    y = 40
    for i, line in enumerate(LINES):
        if not line:
            y += 15
            continue
        font = title_font if i == 0 else body_font
        color = (20, 20, 20) if i != 0 else (10, 60, 140)
        draw.text((40, y), line, fill=color, font=font)
        y += 50 if i == 0 else 38

    image.save(output_path)
    print(f"[OK] Sample image written to: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_sample()
