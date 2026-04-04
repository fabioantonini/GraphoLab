"""
GraphoLab core — Document Layout Detection via PaddleOCR.

Provides:
  - detect_layout()          detect document regions (tables, figures, text)
  - extract_ordered_text()   OCR with reading-order text extraction
  - crop_region()            crop a region bounding box from a document image

Models are lazy-loaded on first call (~185 MB total download on first use).
"""

from __future__ import annotations

import base64
import io
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Lazy model state
# ──────────────────────────────────────────────────────────────────────────────

_layout_engine: Any = None
_ocr_engine: Any = None
_lock = threading.Lock()


def _get_layout():
    """Lazy-load layout detection engine.

    Tries the new PaddleOCR 2.8+ LayoutDetection API first;
    falls back to the stable PPStructure API if not available.
    """
    global _layout_engine
    if _layout_engine is None:
        with _lock:
            if _layout_engine is None:
                try:
                    # PaddleOCR 2.8+ / PaddleX 3.x API
                    from paddleocr import LayoutDetection  # type: ignore
                    _layout_engine = ("new", LayoutDetection())
                except (ImportError, AttributeError):
                    # Fallback: stable PPStructure API (all PaddleOCR versions)
                    from paddleocr import PPStructure  # type: ignore
                    engine = PPStructure(
                        table=False,
                        ocr=False,
                        show_log=False,
                        layout=True,
                    )
                    _layout_engine = ("old", engine)
    return _layout_engine


def _get_ocr():
    """Lazy-load PaddleOCR text recognition engine."""
    global _ocr_engine
    if _ocr_engine is None:
        with _lock:
            if _ocr_engine is None:
                from paddleocr import PaddleOCR  # type: ignore
                # use_angle_cls=True handles rotated text; lang='it' for Italian/European docs
                _ocr_engine = PaddleOCR(
                    use_angle_cls=True,
                    lang="it",
                    show_log=False,
                )
    return _ocr_engine


# ──────────────────────────────────────────────────────────────────────────────
# Internal parsing helpers (normalize both API outputs to the same format)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_new_api(results: Any) -> dict:
    """Parse output from PaddleOCR 2.8+ LayoutDetection.predict()."""
    regions = []
    # results may be a list of result objects or already a flat list of boxes
    if isinstance(results, list):
        raw_boxes = results
    elif isinstance(results, dict):
        raw_boxes = results.get("layout_det_res", {}).get("boxes", [])
    else:
        raw_boxes = []

    # PP-StructureV3 may wrap in an extra dict layer
    if raw_boxes and isinstance(raw_boxes[0], dict) and "layout_det_res" in raw_boxes[0]:
        raw_boxes = raw_boxes[0]["layout_det_res"].get("boxes", [])

    for box in raw_boxes:
        label = box.get("label", box.get("type", "unknown"))
        score = float(box.get("score", box.get("confidence", 0.0)))
        coord = box.get("coordinate", box.get("bbox", []))
        if len(coord) == 4:
            x1, y1, x2, y2 = coord
        elif len(coord) == 8:
            xs = coord[0::2]; ys = coord[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            continue
        regions.append({
            "type": label.lower(),
            "label": label,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "score": score,
        })
    return {"regions": regions}


def _parse_old_api(results: list) -> dict:
    """Parse output from PPStructure() — list of region dicts."""
    regions = []
    for item in results:
        label = item.get("type", "unknown")
        bbox = item.get("bbox", [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
        else:
            continue
        regions.append({
            "type": label.lower(),
            "label": label,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "score": 1.0,
        })
    return {"regions": regions}


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def detect_layout(image_path: str) -> dict:
    """Detect structured regions in a document image.

    Args:
        image_path: Absolute path to the document image (JPG, PNG, PDF page).

    Returns:
        dict with key "regions": list of dicts, each containing:
          - type: str   region category ("text", "table", "figure", "title", ...)
          - label: str  display label
          - bbox: list  [x1, y1, x2, y2] pixel coordinates
          - score: float confidence score
    """
    api_version, layout = _get_layout()
    try:
        if api_version == "new":
            raw = layout.predict(image_path)
            return _parse_new_api(raw)
        else:
            import cv2  # type: ignore
            img = cv2.imread(image_path)
            if img is None:
                return {"regions": [], "error": f"Cannot read image: {image_path}"}
            raw = layout(img)
            return _parse_old_api(raw)
    except Exception as e:
        return {"regions": [], "error": str(e)}


def extract_ordered_text(image_path: str) -> str:
    """Extract text from a document image using OCR, ordered by reading position (top→bottom, left→right).

    Args:
        image_path: Absolute path to the document image.

    Returns:
        Plain text string with lines sorted by vertical then horizontal position.
    """
    ocr = _get_ocr()
    try:
        result = ocr.ocr(image_path, cls=True)
    except Exception as e:
        return f"Errore OCR: {e}"

    if not result or result[0] is None:
        return ""

    lines = []
    for line in result[0]:
        if line is None:
            continue
        # line = [polygon_pts, (text, confidence)]
        if len(line) < 2:
            continue
        polygon = line[0]
        text_conf = line[1]
        if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 1:
            text = str(text_conf[0])
        else:
            continue
        # Use top-left Y coordinate for reading-order sort
        if polygon and len(polygon) >= 1:
            y = polygon[0][1]
            x = polygon[0][0]
        else:
            y, x = 0, 0
        lines.append((y, x, text))

    # Sort: primarily by row (Y), secondarily by column (X)
    lines.sort(key=lambda t: (t[0], t[1]))
    return "\n".join(t[2] for t in lines)


def _image_to_base64(pil_img: Image.Image) -> str:
    """Encode a PIL Image to base64 PNG string for Ollama vision API."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _get_active_model() -> str:
    """Read the currently active model from core.rag (set by the user via sidebar)."""
    try:
        from core.rag import _rag_model
        return _rag_model or "qwen3-vl:8b"
    except Exception:
        return "qwen3-vl:8b"


def call_vision_model(
    pil_img: Image.Image,
    prompt: str,
    model: str | None = None,
    ollama_url: str = "http://localhost:11434",
) -> str:
    """Send an image + prompt to an Ollama multimodal model and return the response.

    Uses the /api/chat endpoint with base64-encoded image — identical to the
    approach shown in the L6 DeepLearning.AI notebook (which used ChatOpenAI;
    here we use the local Ollama equivalent).

    Args:
        pil_img:    PIL Image to analyse.
        prompt:     Text instruction for the model.
        model:      Ollama multimodal model tag (default: qwen3-vl:8b).
        ollama_url: Ollama server URL.

    Returns:
        Model response text, or an error string.
    """
    if model is None:
        model = _get_active_model()
    b64 = _image_to_base64(pil_img)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [b64],
            }
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    try:
        payload["stream"] = True
        r = requests.post(
            f"{ollama_url}/api/chat",
            json=payload,
            stream=True,
            timeout=(10, 300),
        )
        r.raise_for_status()
        content = []
        for line in r.iter_lines():
            if not line:
                continue
            import json as _json
            chunk = _json.loads(line)
            content.append(chunk.get("message", {}).get("content", ""))
            if chunk.get("done"):
                break
        return "".join(content).strip()
    except Exception as e:
        return f"Errore VLM: {e}"


def analyse_table_region(
    image_path: str,
    region_index: int = 0,
    model: str | None = None,
) -> dict:
    """Detect table regions in a document, crop the requested one, analyse with VLM.

    Follows the same pipeline as the L6 notebook:
      detect_layout → crop_region → call_vision_model

    Args:
        image_path:   Absolute path to the document image.
        region_index: Which table to analyse (0 = first detected table).
        model:        Ollama multimodal model to use.

    Returns:
        dict with keys:
          - markdown: str   Markdown table extracted by the VLM
          - region:   dict  Bounding box info of the cropped region
          - error:    str   (only present on failure)
    """
    layout = detect_layout(image_path)
    if "error" in layout:
        return {"error": layout["error"]}

    tables = [r for r in layout["regions"] if r["type"] in ("table", "tabella")]
    if not tables:
        return {"error": "Nessuna tabella rilevata nel documento."}
    if region_index >= len(tables):
        return {"error": f"Indice {region_index} fuori range — trovate {len(tables)} tabelle."}

    region = tables[region_index]
    img = np.array(Image.open(image_path).convert("RGB"))
    crop = crop_region(img, region["bbox"], padding=8)

    prompt = (
        "Sei un assistente forense. Analizza questa tabella estratta da un documento.\n"
        "Estrai tutti i dati in formato Markdown. Mantieni tutte le righe e colonne.\n"
        "Rispondi SOLO con la tabella Markdown, senza testo aggiuntivo."
    )
    markdown = call_vision_model(crop, prompt, model=model)
    return {"markdown": markdown, "region": region}


def analyse_figure_region(
    image_path: str,
    region_index: int = 0,
    model: str | None = None,
) -> dict:
    """Detect figure/chart regions in a document, crop and analyse with VLM.

    Args:
        image_path:   Absolute path to the document image.
        region_index: Which figure to analyse (0 = first detected figure).
        model:        Ollama multimodal model to use.

    Returns:
        dict with keys:
          - description: str  VLM analysis of the figure/chart
          - region:      dict Bounding box info
          - error:       str  (only present on failure)
    """
    layout = detect_layout(image_path)
    if "error" in layout:
        return {"error": layout["error"]}

    figures = [
        r for r in layout["regions"]
        if r["type"] in ("figure", "figura", "chart", "image", "picture", "graph")
    ]
    if not figures:
        return {"error": "Nessuna figura o grafico rilevato nel documento."}
    if region_index >= len(figures):
        return {"error": f"Indice {region_index} fuori range — trovate {len(figures)} figure."}

    region = figures[region_index]
    img = np.array(Image.open(image_path).convert("RGB"))
    crop = crop_region(img, region["bbox"], padding=8)

    prompt = (
        "Sei un assistente forense. Analizza questa figura estratta da un documento.\n"
        "Descrivi in dettaglio: tipo di grafico, assi, valori, trend principali, "
        "dati numerici visibili, legenda. Se è un'immagine generica, descrivila.\n"
        "Rispondi in italiano in modo professionale."
    )
    description = call_vision_model(crop, prompt, model=model)
    return {"description": description, "region": region}


def crop_region(image: np.ndarray, bbox: list, padding: int = 5) -> Image.Image:
    """Crop a bounding-box region from a document image.

    Args:
        image:   RGB numpy array (H, W, 3).
        bbox:    [x1, y1, x2, y2] pixel coordinates.
        padding: Extra pixels to add around the crop.

    Returns:
        PIL Image of the cropped region.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return Image.fromarray(image[y1:y2, x1:x2])
