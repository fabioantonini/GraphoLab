"""
GraphoLab core — Optical Character Recognition (OCR).

Provides:
  - get_trocr()          lazy loader for TrOCR processor + model
  - get_easyocr()        lazy loader for EasyOCR reader (Italian + English)
  - htr_transcribe()     transcribe a handwritten image to text
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

TROCR_MODEL = "microsoft/trocr-large-handwritten"

# ──────────────────────────────────────────────────────────────────────────────
# Lazy model loaders
# ──────────────────────────────────────────────────────────────────────────────

_trocr_processor = None
_trocr_model = None
_trocr_lock = threading.Lock()

_easyocr_reader = None
_easyocr_lock = threading.Lock()


def get_trocr():
    """Return (processor, model) for TrOCR, loading on first call (thread-safe)."""
    global _trocr_processor, _trocr_model
    if _trocr_processor is None:
        with _trocr_lock:
            if _trocr_processor is None:
                import torch
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print("Loading TrOCR...")
                _trocr_processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
                _trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL).to(device)
                _trocr_model.eval()
    return _trocr_processor, _trocr_model


def get_easyocr():
    """Return the EasyOCR reader (Italian + English), loading on first call (thread-safe)."""
    global _easyocr_reader
    if _easyocr_reader is None:
        with _easyocr_lock:
            if _easyocr_reader is None:
                import torch
                import easyocr
                gpu = torch.cuda.is_available()
                print("Loading EasyOCR (Italian)...")
                _easyocr_reader = easyocr.Reader(["it", "en"], gpu=gpu)
    return _easyocr_reader


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _preprocess_for_htr(image: np.ndarray) -> np.ndarray:
    """Deskew + CLAHE contrast enhancement, keeping grayscale gradients for EasyOCR."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Deskew via minAreaRect on ink pixels
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(bw > 0))
    if len(coords) > 100:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        else:
            angle = -angle
        if abs(angle) > 0.3:
            h, w = gray.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            gray = cv2.warpAffine(
                gray, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


# ──────────────────────────────────────────────────────────────────────────────
# Core function
# ──────────────────────────────────────────────────────────────────────────────

def htr_transcribe(image: np.ndarray) -> str:
    """Transcribe a handwritten image to text using EasyOCR.

    Args:
        image: RGB numpy array (H, W, 3) or grayscale (H, W).

    Returns:
        Transcribed text as a string. Returns an error message if image is None.
    """
    if image is None:
        return "Carica un'immagine di testo manoscritto."
    reader = get_easyocr()
    processed = _preprocess_for_htr(image)
    results = reader.readtext(processed, detail=0, paragraph=True)
    return "\n".join(results)
