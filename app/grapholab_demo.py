"""
GraphoLab — Interactive Forensic Graphology Demo
Gradio multi-tab application.

Run:
    python app/grapholab_demo.py

Access:
    http://localhost:7860
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Allow importing from the project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import gradio as gr
from PIL import Image, ImageDraw, ImageOps
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TROCR_MODEL = "microsoft/trocr-base-handwritten"
YOLO_REPO = "tech4humans/yolov8s-signature-detector"
SIG_THRESHOLD = 0.35  # cosine distance threshold for signature verification

# ──────────────────────────────────────────────────────────────────────────────
# Lazy model loaders (loaded on first use to avoid memory duplication)
# ──────────────────────────────────────────────────────────────────────────────

_trocr_processor = None
_trocr_model = None
_yolo_model = None


def get_trocr():
    global _trocr_processor, _trocr_model
    if _trocr_processor is None:
        print("Loading TrOCR...")
        _trocr_processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
        _trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL).to(DEVICE)
        _trocr_model.eval()
    return _trocr_processor, _trocr_model


def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        print("Loading YOLOv8 signature detector...")
        hf_token = os.environ.get("HF_TOKEN")
        # Try common filenames used in YOLO HF repos
        for filename in ("model.pt", "best.pt", "yolov8s.pt"):
            try:
                model_path = hf_hub_download(
                    repo_id=YOLO_REPO, filename=filename, token=hf_token
                )
                _yolo_model = YOLO(model_path)
                print(f"Loaded YOLO model from {filename}")
                break
            except Exception:
                continue
        if _yolo_model is None:
            raise RuntimeError(
                f"Could not find a valid model file in {YOLO_REPO}. "
                "Check the 'Files and versions' tab on HuggingFace for the correct filename."
            )
    return _yolo_model


# ──────────────────────────────────────────────────────────────────────────────
# SigNet Encoder (used for signature verification)
# ──────────────────────────────────────────────────────────────────────────────

class SigNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, 11), nn.ReLU(True),
            nn.LocalResponseNorm(5, 1e-4, 0.75, 2), nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, padding=2), nn.ReLU(True),
            nn.LocalResponseNorm(5, 1e-4, 0.75, 2), nn.MaxPool2d(3, 2), nn.Dropout2d(0.3),
            nn.Conv2d(256, 384, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(3, 2), nn.Dropout2d(0.3),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 17 * 25, 1024), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(1024, 128),
        )

    def forward_once(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return F.normalize(self.fc(x), p=2, dim=1)


_signet = None


def get_signet():
    global _signet
    if _signet is None:
        _signet = SigNetEncoder().to(DEVICE).eval()
    return _signet


SIG_TRANSFORM = T.Compose([
    T.Grayscale(), T.Resize((155, 220)), T.ToTensor(), T.Normalize([0.5], [0.5])
])


def preprocess_signature(pil_img: Image.Image) -> torch.Tensor:
    img = pil_img.convert("RGB")
    arr = np.array(img.convert("L"))
    if arr.mean() < 128:
        img = ImageOps.invert(img)
    return SIG_TRANSFORM(img).unsqueeze(0).to(DEVICE)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 1 — Handwritten OCR
# ──────────────────────────────────────────────────────────────────────────────

def htr_transcribe(image: np.ndarray) -> str:
    if image is None:
        return "Please upload a handwritten image."
    processor, model = get_trocr()
    pil_img = Image.fromarray(image).convert("RGB")
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


htr_tab = gr.Interface(
    fn=htr_transcribe,
    inputs=gr.Image(label="Handwritten text image", type="numpy"),
    outputs=gr.Textbox(label="Transcription", lines=4),
    title="Handwritten Text Recognition (TrOCR)",
    description=(
        "Upload an image of handwritten text. The model will transcribe it automatically.\n\n"
        "**Model:** `microsoft/trocr-base-handwritten` (Hugging Face)\n"
        "**Forensic use:** Anonymous letters, historical documents, court exhibits."
    ),
    examples=[
        [str(ROOT / "data" / "samples" / "handwritten_text_01.png")]
    ] if (ROOT / "data" / "samples" / "handwritten_text_01.png").exists() else [],
    flagging_mode="never",
)

# ──────────────────────────────────────────────────────────────────────────────
# Tab 2 — Signature Verification
# ──────────────────────────────────────────────────────────────────────────────

def sig_verify(ref_image: np.ndarray, query_image: np.ndarray) -> str:
    if ref_image is None or query_image is None:
        return "Please upload both a reference and a questioned signature."
    model = get_signet()
    ref_pil = Image.fromarray(ref_image)
    query_pil = Image.fromarray(query_image)
    t_ref = preprocess_signature(ref_pil)
    t_query = preprocess_signature(query_pil)
    with torch.no_grad():
        emb_ref = model.forward_once(t_ref)
        emb_query = model.forward_once(t_query)
    cosine_sim = F.cosine_similarity(emb_ref, emb_query).item()
    cosine_dist = 1.0 - cosine_sim
    confidence = max(0.0, min(1.0, 1.0 - cosine_dist / 2.0))
    verdict = "GENUINE ✓" if cosine_dist < SIG_THRESHOLD else "FORGED ✗"

    return (
        f"Verdict: {verdict}\n"
        f"Confidence: {confidence:.1%}\n"
        f"Cosine similarity: {cosine_sim:.4f}\n"
        f"Cosine distance: {cosine_dist:.4f}\n"
        f"Threshold: {SIG_THRESHOLD}\n\n"
        f"⚠️  Note: These weights are randomly initialised for demo purposes.\n"
        f"Load pre-trained SigNet weights for production use."
    )


sig_verify_tab = gr.Interface(
    fn=sig_verify,
    inputs=[
        gr.Image(label="Reference signature (known genuine)", type="numpy"),
        gr.Image(label="Questioned signature", type="numpy"),
    ],
    outputs=gr.Textbox(label="Verification result", lines=8),
    title="Signature Authenticity Verification (Siamese Network)",
    description=(
        "Upload a known genuine reference signature and a questioned signature.\n\n"
        "**Technique:** Siamese Neural Network (SigNet)\n"
        "**Forensic use:** Bank cheques, contracts, wills.\n\n"
        "*For production use, load pre-trained weights from [luizgh/sigver](https://github.com/luizgh/sigver).*"
    ),
    flagging_mode="never",
)

# ──────────────────────────────────────────────────────────────────────────────
# Tab 3 — Signature Detection
# ──────────────────────────────────────────────────────────────────────────────

def sig_detect(image: np.ndarray, conf_threshold: float) -> tuple[np.ndarray, str]:
    if image is None:
        return image, "Please upload a document image."
    try:
        yolo = get_yolo()
    except Exception as e:
        msg = (
            "⚠️ **Model not available.**\n\n"
            "The `tech4humans/yolov8s-signature-detector` model is gated on Hugging Face.\n\n"
            "**To enable this tab:**\n"
            "1. Create an account at huggingface.co\n"
            "2. Request access at huggingface.co/tech4humans/yolov8s-signature-detector\n"
            "3. Create a token at huggingface.co/settings/tokens\n"
            "4. Set the environment variable `HF_TOKEN=<your_token>` before starting the app\n\n"
            f"Error: {e}"
        )
        return image, msg
    pil_img = Image.fromarray(image).convert("RGB")

    # Save to temp file for YOLO
    import tempfile
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
        f"Detected {count} signature(s) (confidence ≥ {conf_threshold:.0%})\n\n"
        f"**Model:** `tech4humans/yolov8s-signature-detector`\n"
        f"**Forensic use:** Automated signature extraction from legal documents."
    )
    return annotated, summary


sig_detect_tab = gr.Interface(
    fn=sig_detect,
    inputs=[
        gr.Image(label="Scanned document", type="numpy"),
        gr.Slider(minimum=0.1, maximum=0.9, value=0.3, step=0.05,
                  label="Confidence threshold"),
    ],
    outputs=[
        gr.Image(label="Annotated document", type="numpy"),
        gr.Markdown(label="Detection summary"),
    ],
    title="Signature Detection in Documents (YOLOv8)",
    description=(
        "Upload a scanned document image. The model will locate and highlight all signatures.\n\n"
        "**Model:** YOLOv8 fine-tuned for signature detection\n"
        "**Forensic use:** First step in a detect → extract → verify pipeline."
    ),
    flagging_mode="never",
)

# ──────────────────────────────────────────────────────────────────────────────
# Tab 4 — Graphological Feature Analysis
# ──────────────────────────────────────────────────────────────────────────────

def grapho_analyse(image: np.ndarray) -> tuple[str, np.ndarray]:
    if image is None:
        return "Please upload a handwriting image.", image

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    _, binary = cv2.threshold(
        cv2.GaussianBlur(gray, (3, 3), 0), 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Slant
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= 20 and len(cnt) >= 5:
            _, _, angle = cv2.fitEllipse(cnt)
            slant = angle - 90.0
            if -60 < slant < 60:
                angles.append(slant)
    slant_mean = float(np.mean(angles)) if angles else 0.0
    slant_std = float(np.std(angles)) if angles else 0.0

    # Pressure
    ink_mask = binary > 0
    pressure = (255 - gray)[ink_mask]
    pressure_mean = float(pressure.mean()) if len(pressure) else 0.0

    # Connected components
    num, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    valid = stats[1:][stats[1:, cv2.CC_STAT_AREA] > 15] if num > 1 else np.zeros((0, 5))
    h_mean = float(valid[:, cv2.CC_STAT_HEIGHT].mean()) if len(valid) else 0.0
    w_mean = float(valid[:, cv2.CC_STAT_WIDTH].mean()) if len(valid) else 0.0

    # Word spacing
    h_proj = binary.sum(axis=0)
    gaps = []
    in_gap, gap_w = False, 0
    for v in h_proj:
        if v == 0:
            in_gap = True
            gap_w += 1
        elif in_gap:
            if gap_w > 5:
                gaps.append(gap_w)
            in_gap = False
            gap_w = 0
    word_spacing = float(np.mean(gaps)) if gaps else 0.0

    ink_density = ink_mask.mean() * 100

    # Build annotated visualisation
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    for cnt in contours:
        if cv2.contourArea(cnt) >= 20:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 180, 255), 1)

    report = (
        f"**Graphological Feature Analysis**\n\n"
        f"| Feature | Value |\n"
        f"|---------|-------|\n"
        f"| Mean letter slant | {slant_mean:+.1f}° ({'right' if slant_mean > 0 else 'left' if slant_mean < 0 else 'upright'}) |\n"
        f"| Slant variation (σ) | {slant_std:.1f}° |\n"
        f"| Stroke pressure | {pressure_mean:.1f} / 255 |\n"
        f"| Mean letter height | {h_mean:.1f} px |\n"
        f"| Mean letter width | {w_mean:.1f} px |\n"
        f"| Mean word spacing | {word_spacing:.1f} px |\n"
        f"| Ink density | {ink_density:.2f}% |\n"
        f"| Connected components | {len(valid)} |\n\n"
        f"*Bounding boxes shown in the annotated image.*"
    )
    return report, vis


grapho_tab = gr.Interface(
    fn=grapho_analyse,
    inputs=gr.Image(label="Handwritten text image", type="numpy"),
    outputs=[
        gr.Markdown(label="Analysis report"),
        gr.Image(label="Annotated image (letter bounding boxes)", type="numpy"),
    ],
    title="Graphological Feature Analysis",
    description=(
        "Upload a handwritten text image. The tool extracts graphological metrics "
        "including letter slant, stroke pressure, letter size, and word spacing.\n\n"
        "**Technique:** OpenCV + classical image processing\n"
        "**Forensic use:** Profiling, comparative analysis, expert witness support."
    ),
    flagging_mode="never",
)

# ──────────────────────────────────────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────────────────────────────────────

demo = gr.TabbedInterface(
    interface_list=[htr_tab, sig_verify_tab, sig_detect_tab, grapho_tab],
    tab_names=[
        "Handwritten OCR",
        "Signature Verification",
        "Signature Detection",
        "Graphological Analysis",
    ],
    title="GraphoLab — AI in Forensic Graphology",
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )
