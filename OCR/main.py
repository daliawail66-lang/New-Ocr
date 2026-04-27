"""
Entry point for the OCR application.

Automatically locates the Tesseract binary on Windows and launches the GUI.
"""

from __future__ import annotations

import os
import platform
import sys
from tkinter import messagebox

import pytesseract


WINDOWS_TESSERACT_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    os.path.expandvars(r"%USERPROFILE%\AppData\Local\Tesseract-OCR\tesseract.exe"),
]


def configure_tesseract() -> bool:
    """
    Try to locate a Tesseract binary.

    Returns True if Tesseract is callable, False otherwise.
    """
    if platform.system() == "Windows":
        for path in WINDOWS_TESSERACT_CANDIDATES:
            if path and os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break

    try:
        version = pytesseract.get_tesseract_version()
        print(f"[OK] Tesseract detected (version {version}).")
        return True
    except Exception as exc:
        print(f"[ERROR] Tesseract not found: {exc}")
        return False


def show_tesseract_error() -> None:
    """Display a helpful error message if Tesseract is missing."""
    message = (
        "Tesseract OCR is not installed or not in PATH.\n\n"
        "Please install it from:\n"
        "  https://github.com/UB-Mannheim/tesseract/wiki\n\n"
        "Default install path on Windows:\n"
        "  C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    )
    try:
        messagebox.showerror("Tesseract not found", message)
    except Exception:
        print(message)


def main() -> int:
    if not configure_tesseract():
        show_tesseract_error()
        return 1

    from src.gui import launch_gui

    launch_gui()
    return 0


if __name__ == "__main__":
    sys.exit(main())
