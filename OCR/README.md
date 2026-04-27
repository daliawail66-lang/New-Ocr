# Smart Text Reader — OCR System

**GitHub:** [daliawail66-lang/New-Ocr](https://github.com/daliawail66-lang/New-Ocr)

OCR prototype built in Python for a Master 2 dissertation in
Telecommunications Engineering. Topic: *A text reader for visually impaired
users*. This repository covers **Chapter 1** of the dissertation: a full
detection + recognition pipeline based on OpenCV and Tesseract, with a
professional graphical interface.

## Features

- **Preprocessing** — grayscale, median denoising, Otsu or adaptive binarization
- **Detection** — text localization via Tesseract's layout analysis (bounding boxes + confidence)
- **Recognition** — per-region recognition using Tesseract's LSTM + CTC engine
- **GUI** — dark-mode interface (customtkinter) with three views (Original, Preprocessed, Detection) and live metrics (regions, mean confidence, processing time)

The GitHub copy **does not include** a local `docs/` directory (thesis report scripts and generated PDFs). If you have that folder from the full project, you can run `python docs/generate_report.py` to build the Chapter 1 PDF offline.

## Project structure

```
OCR/
├── main.py                  # Entry point (launches the GUI)
├── generate_sample.py       # Creates a sample test image
├── requirements.txt
├── assets/
│   └── sample.png           # Generated test image
└── src/
    ├── preprocessing.py     # grayscale / denoise / binarize
    ├── detection.py         # text localization
    ├── recognition.py       # Tesseract OCR
    └── gui.py               # customtkinter UI
```

## Installation

### 1. Install Tesseract OCR

Tesseract is a native binary; `pytesseract` is only a Python wrapper around it.

- **Windows**: download the installer from
  https://github.com/UB-Mannheim/tesseract/wiki and install to the default
  path `C:\Program Files\Tesseract-OCR\`. The app auto-detects it.
- **Linux**: `sudo apt install tesseract-ocr`
- **macOS**: `brew install tesseract`

To recognize French or Arabic text, install the matching language packs:
- Windows: select them during installation
- Linux: `sudo apt install tesseract-ocr-fra tesseract-ocr-ara`

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Generate a test image

```bash
python generate_sample.py
```

This creates `assets/sample.png`, a 900×500 image with known text,
useful for a first quick test.

### Launch the GUI

```bash
python main.py
```

Workflow inside the app:

1. **Load Image** — pick any image file
2. **Preprocess** (optional) — see the binary image used by Tesseract
3. **Detect** — see green bounding boxes around every detected text region
4. **Recognize** — see the extracted text
5. **Full Pipeline** — run all three stages at once
6. **Save Text** — export the recognized text to a `.txt` file

The language selector supports English (`eng`), French (`fra`),
Arabic (`ara`) and English + French (`eng+fra`) out of the box.
Additional languages work if the corresponding `.traineddata` file
is installed alongside Tesseract.

## License

Academic work — free for educational and non-commercial use.
