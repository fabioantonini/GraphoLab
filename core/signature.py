"""
GraphoLab core — Signature Verification and Detection.

Provides:
  - get_signet()            lazy loader for the SigNet model
  - get_yolo()              lazy loader for the YOLOv8 signature detector
  - preprocess_signature()  sigver-compatible preprocessing
  - sig_verify()            verify signature authenticity (SigNet)
  - sig_detect()            detect signature locations in a document (YOLO)
  - detect_and_crop()       detect + return annotated image and first crop
"""

from __future__ import annotations

import io
import os
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage import filters, transform as sk_transform

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIGNET_CANVAS = (952, 1360)
SIG_THRESHOLD = 0.35

YOLO_REPO = "tech4humans/yolov8s-signature-detector"
YOLO_FILENAME = "yolov8s.pt"

# ──────────────────────────────────────────────────────────────────────────────
# SigNet architecture
# ──────────────────────────────────────────────────────────────────────────────

def _conv_bn_relu(in_ch, out_ch, kernel, stride=1, pad=0):
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(in_ch, out_ch, kernel, stride, pad, bias=False)),
        ("bn",   nn.BatchNorm2d(out_ch)),
        ("relu", nn.ReLU()),
    ]))


def _linear_bn_relu(in_f, out_f):
    return nn.Sequential(OrderedDict([
        ("fc",   nn.Linear(in_f, out_f, bias=False)),
        ("bn",   nn.BatchNorm1d(out_f)),
        ("relu", nn.ReLU()),
    ]))


class SigNet(nn.Module):
    """SigNet feature extractor (sigver re-implementation, output: 2048-d L2-normalised)."""
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(OrderedDict([
            ("conv1",    _conv_bn_relu(1, 96, 11, stride=4)),
            ("maxpool1", nn.MaxPool2d(3, 2)),
            ("conv2",    _conv_bn_relu(96, 256, 5, pad=2)),
            ("maxpool2", nn.MaxPool2d(3, 2)),
            ("conv3",    _conv_bn_relu(256, 384, 3, pad=1)),
            ("conv4",    _conv_bn_relu(384, 384, 3, pad=1)),
            ("conv5",    _conv_bn_relu(384, 256, 3, pad=1)),
            ("maxpool3", nn.MaxPool2d(3, 2)),
        ]))
        self.fc_layers = nn.Sequential(OrderedDict([
            ("fc1", _linear_bn_relu(256 * 3 * 5, 2048)),
            ("fc2", _linear_bn_relu(2048, 2048)),
        ]))

    def forward_once(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), 256 * 3 * 5)
        x = self.fc_layers(x)
        return F.normalize(x, p=2, dim=1)


# ──────────────────────────────────────────────────────────────────────────────
# Lazy model loaders
# ──────────────────────────────────────────────────────────────────────────────

_signet = None
_signet_pretrained = False
_signet_lock = threading.Lock()

_yolo_model = None
_yolo_lock = threading.Lock()


def get_signet(weights_path: Path):
    """Return the SigNet model, loading weights on first call (thread-safe)."""
    global _signet, _signet_pretrained
    if _signet is None:
        with _signet_lock:
            if _signet is None:
                model = SigNet().to(DEVICE).eval()
                if weights_path.exists():
                    state_dict, _, _ = torch.load(weights_path, map_location=DEVICE)
                    model.load_state_dict(state_dict)
                    _signet_pretrained = True
                    print("SigNet: loaded pre-trained weights from", weights_path)
                else:
                    print("SigNet: no pre-trained weights found — using random initialisation.")
                _signet = model
    return _signet


def get_yolo():
    """Return the YOLO signature detector, downloading on first call (thread-safe)."""
    global _yolo_model
    if _yolo_model is None:
        with _yolo_lock:
            if _yolo_model is None:
                from huggingface_hub import hf_hub_download
                from ultralytics import YOLO
                print("Loading YOLOv8 signature detector...")
                hf_token = os.environ.get("HF_TOKEN")
                model_path = hf_hub_download(
                    repo_id=YOLO_REPO, filename=YOLO_FILENAME, token=hf_token
                )
                _yolo_model = YOLO(model_path)
    return _yolo_model


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_signature(pil_img: Image.Image) -> torch.Tensor:
    """Sigver-compatible preprocessing: centre on canvas, invert, resize to 150×220."""
    arr = np.array(pil_img.convert("L"), dtype=np.uint8)
    canvas = np.ones(SIGNET_CANVAS, dtype=np.uint8) * 255
    try:
        threshold = filters.threshold_otsu(arr)
        blurred = filters.gaussian(arr, 2, preserve_range=True)
        binary = blurred > threshold
        rows, cols = np.where(binary == 0)
        if len(rows) == 0:
            raise ValueError("empty")
        cropped = arr[rows.min():rows.max(), cols.min():cols.max()]
        r_center = int(rows.mean() - rows.min())
        c_center = int(cols.mean() - cols.min())
        r_start = max(0, SIGNET_CANVAS[0] // 2 - r_center)
        c_start = max(0, SIGNET_CANVAS[1] // 2 - c_center)
        h = min(cropped.shape[0], SIGNET_CANVAS[0] - r_start)
        w = min(cropped.shape[1], SIGNET_CANVAS[1] - c_start)
        canvas[r_start:r_start + h, c_start:c_start + w] = cropped[:h, :w]
        canvas[canvas > threshold] = 255
    except Exception:
        canvas = arr

    inverted = 255 - canvas
    resized = sk_transform.resize(inverted, (150, 220), preserve_range=True,
                                  anti_aliasing=True).astype(np.uint8)
    tensor = torch.from_numpy(resized).float().div(255)
    return tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)


# ──────────────────────────────────────────────────────────────────────────────
# Core functions
# ──────────────────────────────────────────────────────────────────────────────

def sig_verify(
    ref_image: np.ndarray,
    ref_image2: np.ndarray | None,
    query_image: np.ndarray,
    weights_path: Path,
) -> tuple[str, np.ndarray | None]:
    """Verify signature authenticity using SigNet.

    Args:
        ref_image:    Known authentic signature (numpy RGB array).
        ref_image2:   Optional second reference (improves accuracy).
        query_image:  Signature to verify (numpy RGB array).
        weights_path: Path to signet.pth weights file.

    Returns:
        report:  Text report with verdict and distances.
        chart:   Matplotlib visualisation as numpy array.
    """
    if ref_image is None or query_image is None:
        return "Carica la firma di riferimento e quella da verificare.", None

    model = get_signet(weights_path)

    with torch.no_grad():
        emb_ref1 = model.forward_once(preprocess_signature(Image.fromarray(ref_image)))
        if ref_image2 is not None:
            emb_ref2 = model.forward_once(preprocess_signature(Image.fromarray(ref_image2)))
            mean_ref = F.normalize(emb_ref1 + emb_ref2, p=2, dim=1)
            n_refs = 2
        else:
            mean_ref = emb_ref1
            n_refs = 1
        emb_query = model.forward_once(preprocess_signature(Image.fromarray(query_image)))

    cosine_sim = F.cosine_similarity(mean_ref, emb_query).item()
    cosine_dist = 1.0 - cosine_sim
    verdict = "AUTENTICA ✓" if cosine_dist < SIG_THRESHOLD else "FALSA ✗"
    color = "#2ca02c" if cosine_dist < SIG_THRESHOLD else "#d62728"

    weights_note = (
        "Modello: SigNet — pesi pre-addestrati GPDS (luizgh/sigver)."
        if _signet_pretrained else
        "⚠️ ATTENZIONE: pesi casuali — risultati non significativi.\n"
        "Scarica signet.pth da luizgh/sigver e posizionalo in models/signet.pth."
    )
    report = (
        f"Esito: {verdict}\n"
        f"Similarità coseno: {cosine_sim:.4f}\n"
        f"Distanza coseno:   {cosine_dist:.4f}  (soglia: {SIG_THRESHOLD})\n"
        f"Riferimenti usati: {n_refs}"
        + (" (embedding mediato)" if n_refs > 1 else "") + "\n\n"
        + weights_note
    )

    # Matplotlib visualisation
    n_img_panels = 2 + (1 if ref_image2 is not None else 0)
    width_ratios = ([1] * n_img_panels) + [1.4]
    fig, axes = plt.subplots(
        1, n_img_panels + 1,
        figsize=(3.2 * (n_img_panels + 1), 3.2),
        gridspec_kw={"width_ratios": width_ratios},
    )
    panels = [ref_image]
    labels = ["Rif. 1"]
    if ref_image2 is not None:
        panels.append(ref_image2)
        labels.append("Rif. 2")
    panels.append(query_image)
    labels.append("Da verificare")

    for ax, img, lbl in zip(axes[:-1], panels, labels):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(lbl, fontsize=10)
        ax.axis("off")

    ax_g = axes[-1]
    ax_g.set_xlim(0, 1)
    ax_g.set_ylim(0, 1)
    ax_g.axis("off")
    ax_g.text(0.5, 0.82, verdict, ha="center", va="center",
              fontsize=14, fontweight="bold", color=color,
              transform=ax_g.transAxes)

    bar_ax = fig.add_axes([
        axes[-1].get_position().x0 + 0.01,
        axes[-1].get_position().y0 + 0.12,
        axes[-1].get_position().width - 0.02,
        0.18,
    ])
    bar_ax.barh([0], [cosine_dist], color=color, alpha=0.75, height=0.6)
    bar_ax.barh([0], [1.0 - cosine_dist], left=cosine_dist,
                color="#cccccc", alpha=0.4, height=0.6)
    bar_ax.axvline(SIG_THRESHOLD, color="black", linestyle="--", linewidth=1.2)
    bar_ax.set_xlim(0, 1)
    bar_ax.set_ylim(-0.5, 0.5)
    bar_ax.set_yticks([])
    bar_ax.set_xticks([0, SIG_THRESHOLD, 1])
    bar_ax.set_xticklabels(["0", f"soglia\n{SIG_THRESHOLD}", "1"], fontsize=7)
    bar_ax.set_xlabel(f"Distanza coseno: {cosine_dist:.3f}", fontsize=8)

    plt.suptitle("Verifica Autenticità Firma — SigNet", fontsize=11, fontweight="bold")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    chart = np.array(Image.open(buf))

    return report, chart


def sig_detect(
    image: np.ndarray,
    conf_threshold: float,
) -> tuple[np.ndarray, str]:
    """Detect signature locations in a document image using YOLO.

    Args:
        image:          RGB numpy array of the document.
        conf_threshold: Confidence threshold (0.1–0.9).

    Returns:
        annotated:  Image with bounding boxes drawn.
        summary:    Markdown summary string.
    """
    if image is None:
        return image, "Carica un'immagine del documento."
    try:
        yolo = get_yolo()
    except Exception as e:
        msg = (
            "⚠️ **Modello non disponibile.**\n\n"
            "Il modello `tech4humans/yolov8s-signature-detector` è ad accesso limitato su Hugging Face.\n\n"
            "**Per abilitare questa sezione:**\n"
            "1. Crea un account su huggingface.co\n"
            "2. Richiedi l'accesso su huggingface.co/tech4humans/yolov8s-signature-detector\n"
            "3. Crea un token su huggingface.co/settings/tokens\n"
            "4. Imposta la variabile d'ambiente `HF_TOKEN=<il_tuo_token>` prima di avviare l'app\n\n"
            f"Errore: {e}"
        )
        return image, msg

    pil_img = Image.fromarray(image).convert("RGB")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pil_img.save(tmp.name)
        tmp_path = tmp.name

    results = yolo.predict(tmp_path, conf=conf_threshold, verbose=False)
    os.unlink(tmp_path)

    result = results[0]
    annotated = image.copy()
    count = 0

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu())
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated, f"Sig #{count+1}  {conf:.0%}",
                        (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            count += 1

    summary = (
        f"Rilevat{'a' if count == 1 else 'e'} {count} firma{'' if count == 1 else 'e'} "
        f"(confidenza ≥ {conf_threshold:.0%})\n\n"
        f"**Modello:** `tech4humans/yolov8s-signature-detector`\n"
        f"**Uso forense:** Estrazione automatica di firme da documenti legali."
    )
    return annotated, summary


def detect_and_crop(
    image: np.ndarray,
    conf_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Run YOLO detection and return (annotated, first_crop, summary).

    Gracefully degrades when YOLO is not available (missing HF_TOKEN).
    """
    annotated = image.copy()
    try:
        yolo = get_yolo()
    except Exception:
        return annotated, None, "⚠️ Rilevamento firma non disponibile (HF_TOKEN mancante)."

    pil_img = Image.fromarray(image).convert("RGB")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pil_img.save(tmp.name)
        tmp_path = tmp.name

    results = yolo.predict(tmp_path, conf=conf_threshold, verbose=False)
    os.unlink(tmp_path)

    result = results[0]
    first_crop: np.ndarray | None = None
    count = 0

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu())
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated, f"Sig #{count+1}  {conf:.0%}",
                        (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            if count == 0:
                x1c = max(0, x1); y1c = max(0, y1)
                x2c = min(image.shape[1], x2); y2c = min(image.shape[0], y2)
                if x2c > x1c and y2c > y1c:
                    first_crop = image[y1c:y2c, x1c:x2c]
            count += 1

    summary = (
        f"Rilevat{'a' if count == 1 else 'e'} {count} firma{'' if count == 1 else 'e'}."
        if count > 0
        else "Nessuna firma rilevata nel documento."
    )
    return annotated, first_crop, summary
