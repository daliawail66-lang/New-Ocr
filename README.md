# New-Ocr — Smart Text Reader (OCR)

Python OCR prototype: detection and recognition with OpenCV and Tesseract, plus a **customtkinter** GUI. Built for a Master 2 thesis on assistive text reading for visually impaired users.

**Repository:** [https://github.com/daliawail66-lang/New-Ocr](https://github.com/daliawail66-lang/New-Ocr)

## Quick start

1. **Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)** (required; `pytesseract` is only a wrapper). On Windows, a typical path is `C:\Program Files\Tesseract-OCR\`.

2. **Python dependencies** (from the `OCR` project folder):

   ```bash
   cd OCR
   pip install -r requirements.txt
   ```

3. **Run the app:**

   ```bash
   python main.py
   ```

4. Optional: generate a test image with `python generate_sample.py`.

## Documentation

Full feature list, project layout, language packs (e.g. French, Arabic), and PDF report details are in **[OCR/README.md](OCR/README.md).**

## License

Academic work — free for educational and non-commercial use (see [OCR/README.md](OCR/README.md)).
