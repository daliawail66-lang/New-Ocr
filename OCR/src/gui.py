"""
Graphical user interface for the OCR system.

Built with customtkinter for a modern, dark-mode look. The GUI exposes the
three stages of the pipeline (preprocess, detect, recognize) as separate
buttons so the user can inspect the output of each stage -- useful both
as a demonstration tool and as a screenshot source for the dissertation.
"""

from __future__ import annotations

import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List, Optional

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from .detection import (
    TextRegion,
    detect_text_regions,
    detection_stats,
    draw_bounding_boxes,
)
from .preprocessing import preprocess, to_grayscale
from .recognition import (
    RecognitionResult,
    recognize_full_image,
    recognize_regions,
    results_to_text,
)


DISPLAY_W = 720
DISPLAY_H = 520


class OCRApp(ctk.CTk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Smart Text Reader - OCR System (Master 2 Dissertation)")
        self.geometry("1400x860")
        self.minsize(1200, 760)

        self.original_image: Optional[np.ndarray] = None
        self.gray_image: Optional[np.ndarray] = None
        self.preprocessed_image: Optional[np.ndarray] = None
        self.displayed_image: Optional[np.ndarray] = None
        self.regions: List[TextRegion] = []
        self.results: List[RecognitionResult] = []
        self.current_image_path: Optional[str] = None
        self._tk_image: Optional[ImageTk.PhotoImage] = None

        self._build_ui()
        self._set_status("Ready. Load an image to begin.")

    # --------------------------------------------------------------
    # UI construction
    # --------------------------------------------------------------
    def _build_ui(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(15, 5))

        ctk.CTkLabel(
            header,
            text="Smart Text Reader",
            font=ctk.CTkFont(size=24, weight="bold"),
        ).pack(side="left")

        ctk.CTkLabel(
            header,
            text="OCR System Simulation  |  Detection + Recognition",
            font=ctk.CTkFont(size=13),
            text_color="gray70",
        ).pack(side="left", padx=15)

        content = ctk.CTkFrame(self)
        content.pack(fill="both", expand=True, padx=20, pady=10)
        content.grid_columnconfigure(0, weight=3)
        content.grid_columnconfigure(1, weight=2)
        content.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(content)
        left.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        self._build_left_panel(left)

        right = ctk.CTkFrame(content)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        self._build_right_panel(right)

        self.status = ctk.CTkLabel(
            self,
            text="Ready",
            anchor="w",
            font=ctk.CTkFont(size=11),
            text_color="gray60",
        )
        self.status.pack(fill="x", padx=20, pady=(0, 10))

    def _build_left_panel(self, parent: ctk.CTkFrame) -> None:
        ctk.CTkLabel(
            parent,
            text="Image View",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(15, 5))

        self.view_tabs = ctk.CTkSegmentedButton(
            parent,
            values=["Original", "Preprocessed", "Detection"],
            command=self._on_view_change,
        )
        self.view_tabs.set("Original")
        self.view_tabs.pack(pady=5)

        self.canvas = tk.Canvas(
            parent,
            width=DISPLAY_W,
            height=DISPLAY_H,
            bg="#1a1a1a",
            highlightthickness=0,
        )
        self.canvas.pack(padx=15, pady=10, expand=True)
        self.canvas.create_text(
            DISPLAY_W // 2,
            DISPLAY_H // 2,
            text="No image loaded",
            fill="gray",
            font=("Arial", 14),
        )

        ctrl = ctk.CTkFrame(parent, fg_color="transparent")
        ctrl.pack(pady=(5, 15))

        self._make_button(ctrl, "Load Image", self.load_image).pack(
            side="left", padx=4
        )
        self._make_button(ctrl, "Preprocess", self.do_preprocess).pack(
            side="left", padx=4
        )
        self._make_button(ctrl, "Detect", self.do_detect).pack(
            side="left", padx=4
        )
        self._make_button(ctrl, "Recognize", self.do_recognize).pack(
            side="left", padx=4
        )
        self._make_button(
            ctrl,
            "Full Pipeline",
            self.do_full_pipeline,
            fg_color="#1f8a3a",
            hover_color="#166b2c",
        ).pack(side="left", padx=4)

    def _build_right_panel(self, parent: ctk.CTkFrame) -> None:
        ctk.CTkLabel(
            parent,
            text="Recognized Text",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(15, 5))

        self.text_box = ctk.CTkTextbox(
            parent,
            font=ctk.CTkFont(family="Consolas", size=13),
            wrap="word",
        )
        self.text_box.pack(fill="both", expand=True, padx=15, pady=10)

        metrics_frame = ctk.CTkFrame(parent)
        metrics_frame.pack(fill="x", padx=15, pady=(0, 10))

        self.metric_regions = self._make_metric_row(
            metrics_frame, "Regions detected", "0"
        )
        self.metric_confidence = self._make_metric_row(
            metrics_frame, "Mean confidence", "0.0 %"
        )
        self.metric_time = self._make_metric_row(
            metrics_frame, "Processing time", "0 ms"
        )

        ctk.CTkLabel(
            parent,
            text="Language",
            font=ctk.CTkFont(size=12),
        ).pack(pady=(10, 0))

        self.lang_var = ctk.StringVar(value="eng")
        ctk.CTkOptionMenu(
            parent,
            variable=self.lang_var,
            values=["eng", "fra", "ara", "eng+fra"],
            width=150,
        ).pack(pady=5)

        actions = ctk.CTkFrame(parent, fg_color="transparent")
        actions.pack(fill="x", padx=15, pady=10)

        self._make_button(actions, "Save Text", self.save_text).pack(
            side="left", padx=4, expand=True, fill="x"
        )
        self._make_button(
            actions,
            "Clear",
            self.clear,
            fg_color="gray40",
            hover_color="gray30",
        ).pack(side="left", padx=4, expand=True, fill="x")

    def _make_button(
        self,
        parent,
        text: str,
        command,
        fg_color: Optional[str] = None,
        hover_color: Optional[str] = None,
    ) -> ctk.CTkButton:
        kwargs = {"text": text, "command": command, "width": 120, "height": 34}
        if fg_color:
            kwargs["fg_color"] = fg_color
        if hover_color:
            kwargs["hover_color"] = hover_color
        return ctk.CTkButton(parent, **kwargs)

    def _make_metric_row(
        self, parent: ctk.CTkFrame, label: str, initial: str
    ) -> ctk.CTkLabel:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=3)
        ctk.CTkLabel(
            row,
            text=label + ":",
            font=ctk.CTkFont(size=12),
            text_color="gray70",
            anchor="w",
        ).pack(side="left", padx=10)
        value = ctk.CTkLabel(
            row,
            text=initial,
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="e",
        )
        value.pack(side="right", padx=10)
        return value

    # --------------------------------------------------------------
    # Actions
    # --------------------------------------------------------------
    def load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Error", f"Could not read image:\n{path}")
            return

        self.current_image_path = path
        self.original_image = image
        self.gray_image = to_grayscale(image)
        self.preprocessed_image = None
        self.regions = []
        self.results = []
        self.text_box.delete("1.0", "end")
        self.view_tabs.set("Original")
        self._show_image(self.original_image)
        self._set_status(f"Loaded: {os.path.basename(path)}")

    def do_preprocess(self) -> None:
        if not self._require_image():
            return
        t0 = time.perf_counter()
        self.preprocessed_image = preprocess(self.original_image)
        dt = (time.perf_counter() - t0) * 1000
        self.view_tabs.set("Preprocessed")
        self._show_image(self.preprocessed_image)
        self._update_time(dt)
        self._set_status(f"Preprocessing done in {dt:.0f} ms")

    def do_detect(self) -> None:
        if not self._require_image():
            return
        if self.preprocessed_image is None:
            self.preprocessed_image = preprocess(self.original_image)

        t0 = time.perf_counter()
        self.regions = detect_text_regions(
            self.gray_image, lang=self.lang_var.get()
        )
        dt = (time.perf_counter() - t0) * 1000

        boxed = draw_bounding_boxes(self.original_image, self.regions)
        self.view_tabs.set("Detection")
        self._show_image(boxed)

        stats = detection_stats(self.regions)
        self.metric_regions.configure(text=str(stats["count"]))
        self.metric_confidence.configure(
            text=f'{stats["mean_confidence"]:.1f} %'
        )
        self._update_time(dt)
        self._set_status(
            f'Detection done: {stats["count"]} regions in {dt:.0f} ms'
        )

    def do_recognize(self) -> None:
        if not self._require_image():
            return
        if not self.regions:
            self.do_detect()

        t0 = time.perf_counter()
        self.results = recognize_regions(
            self.gray_image,
            self.regions,
            lang=self.lang_var.get(),
        )
        text = recognize_full_image(
            self.gray_image, lang=self.lang_var.get(), psm=3
        )
        if not text:
            text = results_to_text(self.results)
        dt = (time.perf_counter() - t0) * 1000

        self.text_box.delete("1.0", "end")
        self.text_box.insert("1.0", text if text else "(no text recognized)")
        self._update_time(dt)
        self._set_status(f"Recognition done in {dt:.0f} ms")

    def do_full_pipeline(self) -> None:
        if not self._require_image():
            return
        t0 = time.perf_counter()

        self.preprocessed_image = preprocess(self.original_image)
        self.regions = detect_text_regions(
            self.gray_image, lang=self.lang_var.get()
        )
        self.results = recognize_regions(
            self.gray_image,
            self.regions,
            lang=self.lang_var.get(),
        )
        text = recognize_full_image(
            self.gray_image, lang=self.lang_var.get(), psm=3
        )
        if not text:
            text = results_to_text(self.results)

        dt = (time.perf_counter() - t0) * 1000

        boxed = draw_bounding_boxes(self.original_image, self.regions)
        self.view_tabs.set("Detection")
        self._show_image(boxed)

        stats = detection_stats(self.regions)
        self.metric_regions.configure(text=str(stats["count"]))
        self.metric_confidence.configure(
            text=f'{stats["mean_confidence"]:.1f} %'
        )
        self._update_time(dt)

        self.text_box.delete("1.0", "end")
        self.text_box.insert("1.0", text if text else "(no text recognized)")
        self._set_status(
            f'Full pipeline done in {dt:.0f} ms '
            f'({stats["count"]} regions)'
        )

    def save_text(self) -> None:
        text = self.text_box.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Nothing to save", "No text to save.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt"), ("All files", "*.*")],
            title="Save recognized text",
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        self._set_status(f"Saved to {os.path.basename(path)}")

    def clear(self) -> None:
        self.original_image = None
        self.gray_image = None
        self.preprocessed_image = None
        self.regions = []
        self.results = []
        self.current_image_path = None
        self.text_box.delete("1.0", "end")
        self.canvas.delete("all")
        self.canvas.create_text(
            DISPLAY_W // 2,
            DISPLAY_H // 2,
            text="No image loaded",
            fill="gray",
            font=("Arial", 14),
        )
        self.metric_regions.configure(text="0")
        self.metric_confidence.configure(text="0.0 %")
        self.metric_time.configure(text="0 ms")
        self._set_status("Cleared.")

    # --------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------
    def _on_view_change(self, value: str) -> None:
        if value == "Original" and self.original_image is not None:
            self._show_image(self.original_image)
        elif value == "Preprocessed" and self.preprocessed_image is not None:
            self._show_image(self.preprocessed_image)
        elif value == "Detection" and self.original_image is not None:
            if self.regions:
                self._show_image(
                    draw_bounding_boxes(self.original_image, self.regions)
                )
            else:
                self._show_image(self.original_image)

    def _show_image(self, image: np.ndarray) -> None:
        if len(image.shape) == 2:
            display = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = display.shape[:2]
        scale = min(DISPLAY_W / w, DISPLAY_H / h, 1.0)
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            display = cv2.resize(display, new_size, interpolation=cv2.INTER_AREA)

        pil = Image.fromarray(display)
        self._tk_image = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(
            DISPLAY_W // 2, DISPLAY_H // 2, image=self._tk_image
        )

    def _require_image(self) -> bool:
        if self.original_image is None:
            messagebox.showwarning(
                "No image", "Please load an image first."
            )
            return False
        return True

    def _update_time(self, ms: float) -> None:
        self.metric_time.configure(text=f"{ms:.0f} ms")

    def _set_status(self, text: str) -> None:
        self.status.configure(text=text)


def launch_gui() -> None:
    """Start the application's event loop."""
    app = OCRApp()
    app.mainloop()
