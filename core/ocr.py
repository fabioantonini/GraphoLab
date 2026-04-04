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

# Active OCR model — set via set_ocr_model() / sidebar selector
# Options: "easyocr" | "vlm" | "paddleocr" | "trocr"
def _load_ocr_model_from_env() -> str:
    import os
    val = os.environ.get("OCR_MODEL", "").strip().lower()
    if val in {"easyocr", "vlm", "paddleocr", "trocr"}:
        return val
    try:
        from pathlib import Path
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("OCR_MODEL="):
                    v = line.split("=", 1)[1].strip().lower()
                    if v in {"easyocr", "vlm", "paddleocr", "trocr"}:
                        return v
    except Exception:
        pass
    return "easyocr"

_ocr_model: str = _load_ocr_model_from_env()


def get_ocr_model() -> str:
    return _ocr_model


def set_ocr_model(model: str) -> str:
    global _ocr_model
    allowed = {"easyocr", "vlm", "paddleocr", "trocr"}
    if model not in allowed:
        return f"❌ Modello non valido. Scegli tra: {', '.join(sorted(allowed))}"
    _ocr_model = model
    _persist_ocr_model(model)
    return f"✅ Modello OCR: **{_ocr_model}**"


def _persist_ocr_model(model: str) -> None:
    """Write OCR_MODEL=<model> to .env for persistence across restarts."""
    from pathlib import Path as _Path
    env_file = _Path(__file__).parent.parent / ".env"
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines() if env_file.exists() else []
        found = False
        for i, line in enumerate(lines):
            if line.startswith("OCR_MODEL="):
                lines[i] = f"OCR_MODEL={model}"
                found = True
                break
        if not found:
            lines.append(f"OCR_MODEL={model}")
        env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        pass

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

_HTR_PROMPT = (
    "Sei un esperto paleografo forense. Trascrivi FEDELMENTE tutto il testo "
    "presente in questa immagine, incluso testo manoscritto, stampato o misto.\n"
    "- Mantieni la struttura del documento (paragrafi, a capo, elenchi).\n"
    "- Se una parola è illeggibile scrivi [illeggibile].\n"
    "- NON aggiungere commenti o spiegazioni: rispondi SOLO con il testo trascritto."
)


def _vlm_transcribe(image: np.ndarray, ollama_url: str = "http://localhost:11434") -> str:
    """Transcribe via qwen3-vl:8b (Ollama) using streaming API.

    Uses stream=True so the HTTP connection stays alive token-by-token,
    avoiding read timeouts on long documents.
    Raises on any failure.
    """
    import base64
    import io
    import json
    import requests
    from PIL import Image as _PILImage

    if image.ndim == 2:
        pil_img = _PILImage.fromarray(image).convert("RGB")
    else:
        pil_img = _PILImage.fromarray(image)

    # Resize to max 1500px on the longer side to keep inference fast
    max_side = 1500
    w, h = pil_img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), _PILImage.LANCZOS)

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Use the globally selected model if set, else hardcoded qwen3-vl:8b
    try:
        from core.rag import _rag_model
        model = _rag_model or "qwen3-vl:8b"
    except Exception:
        model = "qwen3-vl:8b"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _HTR_PROMPT, "images": [b64]}],
        "stream": True,
        "options": {"temperature": 0},
    }
    # stream=True: each line is a JSON chunk; connection stays alive per token
    r = requests.post(
        f"{ollama_url}/api/chat",
        json=payload,
        stream=True,
        timeout=(10, 300),  # (connect timeout, read timeout between chunks)
    )
    r.raise_for_status()
    content = []
    for line in r.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        content.append(chunk.get("message", {}).get("content", ""))
        if chunk.get("done"):
            break
    return "".join(content).strip()


def htr_transcribe(image: np.ndarray) -> str:
    """Transcribe a handwritten image to text using the active OCR model.

    The active model is controlled by set_ocr_model() / sidebar selector:
      - "easyocr"   : EasyOCR (default, fast, good for printed+handwritten)
      - "vlm"       : qwen3-vl via Ollama (best for cursive Italian)
      - "paddleocr" : PaddleOCR (good for mixed documents)
      - "trocr"     : Microsoft TrOCR large handwritten

    Args:
        image: RGB numpy array (H, W, 3) or grayscale (H, W).
    """
    if image is None:
        return "Carica un'immagine di testo manoscritto."

    model = _ocr_model

    if model == "vlm":
        try:
            return _vlm_transcribe(image)
        except Exception as e:
            return f"Errore VLM: {e}"

    if model == "paddleocr":
        try:
            from core.document_layout import extract_ordered_text as _paddle_ocr
            import tempfile, os
            from PIL import Image as _PILImage
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            _PILImage.fromarray(image).save(tmp.name)
            tmp.close()
            result = _paddle_ocr(tmp.name)
            os.unlink(tmp.name)
            return result
        except Exception as e:
            return f"Errore PaddleOCR: {e}"

    if model == "trocr":
        try:
            import torch
            from PIL import Image as _PILImage
            processor, trocr_model = get_trocr()
            pil_img = _PILImage.fromarray(image).convert("RGB")
            pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
            device = next(trocr_model.parameters()).device
            pixel_values = pixel_values.to(device)
            with torch.no_grad():
                ids = trocr_model.generate(pixel_values)
            return processor.batch_decode(ids, skip_special_tokens=True)[0]
        except Exception as e:
            return f"Errore TrOCR: {e}"

    # Default: EasyOCR — read raw RGB, no preprocessing
    reader = get_easyocr()
    results = reader.readtext(image, detail=0, paragraph=True)
    return "\n".join(results)
