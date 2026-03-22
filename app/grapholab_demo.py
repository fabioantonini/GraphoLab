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
from collections import OrderedDict
from PIL import Image, ImageDraw, ImageOps
from skimage import filters, transform as sk_transform
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, pipeline as hf_pipeline
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TROCR_MODEL = "microsoft/trocr-base-handwritten"
YOLO_REPO = "tech4humans/yolov8s-signature-detector"
YOLO_FILENAME = "yolov8s.pt"
SIG_THRESHOLD = 0.35  # cosine distance threshold for signature verification
NER_MODEL = "Babelscape/wikineural-multilingual-ner"

# ──────────────────────────────────────────────────────────────────────────────
# Lazy model loaders (loaded on first use to avoid memory duplication)
# ──────────────────────────────────────────────────────────────────────────────

_trocr_processor = None
_trocr_model = None
_yolo_model = None
_ner_pipeline = None


def get_trocr():
    global _trocr_processor, _trocr_model
    if _trocr_processor is None:
        print("Loading TrOCR...")
        _trocr_processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
        _trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL).to(DEVICE)
        _trocr_model.eval()
    return _trocr_processor, _trocr_model


def get_ner():
    global _ner_pipeline
    if _ner_pipeline is None:
        print("Loading NER model...")
        _ner_pipeline = hf_pipeline(
            "ner",
            model=NER_MODEL,
            aggregation_strategy="simple",
            device=0 if DEVICE == "cuda" else -1,
        )
    return _ner_pipeline


def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        print("Loading YOLOv8 signature detector...")
        hf_token = os.environ.get("HF_TOKEN")
        model_path = hf_hub_download(
            repo_id=YOLO_REPO, filename=YOLO_FILENAME, token=hf_token
        )
        _yolo_model = YOLO(model_path)
    return _yolo_model


# ──────────────────────────────────────────────────────────────────────────────
# SigNet (sigver architecture — Hafemann et al. 2017)
# ──────────────────────────────────────────────────────────────────────────────

SIGNET_WEIGHTS = ROOT / "models" / "signet.pth"
SIGNET_CANVAS  = (952, 1360)   # max signature canvas for preprocessing


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


_signet = None
_signet_pretrained = False


def get_signet():
    global _signet, _signet_pretrained
    if _signet is None:
        model = SigNet().to(DEVICE).eval()
        if SIGNET_WEIGHTS.exists():
            state_dict, _, _ = torch.load(SIGNET_WEIGHTS, map_location=DEVICE)
            model.load_state_dict(state_dict)
            _signet_pretrained = True
            print("SigNet: loaded pre-trained weights from", SIGNET_WEIGHTS)
        else:
            print("SigNet: no pre-trained weights found — using random initialisation.")
        _signet = model
    return _signet


def preprocess_signature(pil_img: Image.Image) -> torch.Tensor:
    """Sigver-compatible preprocessing: centre on canvas, invert, resize to 150×220."""
    arr = np.array(pil_img.convert("L"), dtype=np.uint8)

    # Centre on canvas
    canvas = np.ones(SIGNET_CANVAS, dtype=np.uint8) * 255
    try:
        threshold = filters.threshold_otsu(arr)
        blurred   = filters.gaussian(arr, 2, preserve_range=True)
        binary    = blurred > threshold
        rows, cols = np.where(binary == 0)
        if len(rows) == 0:
            raise ValueError("empty")
        cropped   = arr[rows.min():rows.max(), cols.min():cols.max()]
        r_center  = int(rows.mean() - rows.min())
        c_center  = int(cols.mean() - cols.min())
        r_start   = max(0, SIGNET_CANVAS[0] // 2 - r_center)
        c_start   = max(0, SIGNET_CANVAS[1] // 2 - c_center)
        h = min(cropped.shape[0], SIGNET_CANVAS[0] - r_start)
        w = min(cropped.shape[1], SIGNET_CANVAS[1] - c_start)
        canvas[r_start:r_start + h, c_start:c_start + w] = cropped[:h, :w]
        canvas[canvas > threshold] = 255
    except Exception:
        canvas = arr  # fallback: use image as-is

    # Invert and resize to 150×220
    inverted = 255 - canvas
    resized  = sk_transform.resize(inverted, (150, 220), preserve_range=True,
                                   anti_aliasing=True).astype(np.uint8)
    tensor   = torch.from_numpy(resized).float().div(255)
    return tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1, 1, 150, 220)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 1 — Handwritten OCR
# ──────────────────────────────────────────────────────────────────────────────

def _segment_lines(pil_img: Image.Image, min_gap: int = 5, pad: int = 6) -> list[Image.Image]:
    """Split a multi-line handwritten image into individual line crops."""
    gray = np.array(pil_img.convert("L"))
    _, binary = cv2.threshold(
        cv2.GaussianBlur(gray, (3, 3), 0), 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    # Horizontal projection: ink pixels per row
    proj = binary.sum(axis=1)
    in_line, start, segments = False, 0, []
    for r, v in enumerate(proj):
        if v > 0 and not in_line:
            in_line, start = True, r
        elif v == 0 and in_line:
            if r - start >= min_gap:
                segments.append((start, r))
            in_line = False
    if in_line:
        segments.append((start, len(proj)))
    if not segments:
        return [pil_img]
    h, w = gray.shape
    return [
        pil_img.crop((0, max(0, y0 - pad), w, min(h, y1 + pad)))
        for y0, y1 in segments
    ]


def htr_transcribe(image: np.ndarray) -> str:
    if image is None:
        return "Carica un'immagine di testo manoscritto."
    processor, model = get_trocr()
    pil_img = Image.fromarray(image).convert("RGB")
    lines = _segment_lines(pil_img)
    texts = []
    for line_img in lines:
        pixel_values = processor(images=line_img, return_tensors="pt").pixel_values.to(DEVICE)
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        texts.append(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
    return "\n".join(texts)


htr_tab = gr.Interface(
    fn=htr_transcribe,
    inputs=gr.Image(label="Immagine di testo manoscritto", type="numpy"),
    outputs=gr.Textbox(label="Trascrizione", lines=8),
    title="Riconoscimento Testo Manoscritto (TrOCR)",
    description=(
        "Carica un'immagine di testo scritto a mano. Il modello la trascriverà automaticamente.\n\n"
        "**Modello:** `microsoft/trocr-base-handwritten` (Hugging Face)\n"
        "**Uso forense:** Lettere anonime, documenti storici, atti processuali."
    ),
    examples=[
        [str(ROOT / "data" / "samples" / "handwritten_text_01.png")],
        [str(ROOT / "data" / "samples" / "handwritten_multiline_01.png")],
    ] if (ROOT / "data" / "samples" / "handwritten_text_01.png").exists() else [],
    flagging_mode="never",
)

# ──────────────────────────────────────────────────────────────────────────────
# Tab 2 — Signature Verification
# ──────────────────────────────────────────────────────────────────────────────

def sig_verify(ref_image: np.ndarray, query_image: np.ndarray) -> str:
    if ref_image is None or query_image is None:
        return "Carica entrambe le immagini: la firma di riferimento e quella da verificare."
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
    verdict = "AUTENTICA ✓" if cosine_dist < SIG_THRESHOLD else "FALSA ✗"

    weights_note = (
        "Pesi SigNet pre-addestrati caricati (luizgh/sigver — dataset GPDS)."
        if _signet_pretrained else
        "ATTENZIONE: pesi casuali — i risultati non sono significativi.\n"
        "Scarica signet.pth da luizgh/sigver e posizionalo in models/signet.pth."
    )
    return (
        f"Esito: {verdict}\n"
        f"Confidenza: {confidence:.1%}\n"
        f"Similarità coseno: {cosine_sim:.4f}\n"
        f"Distanza coseno: {cosine_dist:.4f}\n"
        f"Soglia: {SIG_THRESHOLD}\n\n"
        f"{weights_note}"
    )


sig_verify_tab = gr.Interface(
    fn=sig_verify,
    inputs=[
        gr.Image(label="Firma di riferimento (autentica nota)", type="numpy"),
        gr.Image(label="Firma da verificare", type="numpy"),
    ],
    outputs=gr.Textbox(label="Risultato della verifica", lines=8),
    title="Verifica Autenticità Firma (Rete Siamese)",
    description=(
        "Carica una firma autentica di riferimento e una firma da verificare.\n\n"
        "**Tecnica:** Rete Neurale Siamese (SigNet)\n"
        "**Uso forense:** Assegni bancari, contratti, testamenti.\n\n"
        "*Per uso in produzione, caricare i pesi pre-addestrati da [luizgh/sigver](https://github.com/luizgh/sigver).*"
    ),
    flagging_mode="never",
)

# ──────────────────────────────────────────────────────────────────────────────
# Tab 3 — Signature Detection
# ──────────────────────────────────────────────────────────────────────────────

def sig_detect(image: np.ndarray, conf_threshold: float) -> tuple[np.ndarray, str]:
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
        f"Rilevat{'a' if count == 1 else 'e'} {count} firma{'' if count == 1 else 'e'} "
        f"(confidenza ≥ {conf_threshold:.0%})\n\n"
        f"**Modello:** `tech4humans/yolov8s-signature-detector`\n"
        f"**Uso forense:** Estrazione automatica di firme da documenti legali."
    )
    return annotated, summary


sig_detect_tab = gr.Interface(
    fn=sig_detect,
    inputs=[
        gr.Image(label="Documento scansionato", type="numpy"),
        gr.Slider(minimum=0.1, maximum=0.9, value=0.3, step=0.05,
                  label="Soglia di confidenza"),
    ],
    outputs=[
        gr.Image(label="Documento annotato", type="numpy"),
        gr.Markdown(label="Riepilogo rilevamento"),
    ],
    title="Rilevamento Firme nei Documenti (YOLOv8)",
    description=(
        "Carica un'immagine di un documento scansionato. Il modello individuerà ed evidenzierà tutte le firme presenti.\n\n"
        "**Modello:** YOLOv8 ottimizzato per il rilevamento di firme\n"
        "**Uso forense:** Primo passo nella pipeline rileva → estrai → verifica."
    ),
    flagging_mode="never",
)

# ──────────────────────────────────────────────────────────────────────────────
# Tab 4 — Named Entity Recognition
# ──────────────────────────────────────────────────────────────────────────────

_NER_LABELS = {"PER": "Persona", "ORG": "Organizzazione", "LOC": "Luogo", "MISC": "Varie"}


def ner_extract(text: str):
    if not text or not text.strip():
        return [], "Inserisci del testo da analizzare."
    nlp = get_ner()
    entities = nlp(text)

    # Build HighlightedText format: list of (span, label|None)
    result = []
    prev_end = 0
    for ent in entities:
        start, end = ent["start"], ent["end"]
        if start > prev_end:
            result.append((text[prev_end:start], None))
        result.append((text[start:end], ent["entity_group"]))
        prev_end = end
    if prev_end < len(text):
        result.append((text[prev_end:], None))

    # Summary table
    if entities:
        rows = "\n".join(
            f"| **{_NER_LABELS.get(e['entity_group'], e['entity_group'])}** "
            f"(`{e['entity_group']}`) | {e['word']} | {e['score']:.0%} |"
            for e in entities
        )
        summary = f"| Tipo | Entità | Confidenza |\n|------|--------|------------|\n{rows}"
    else:
        summary = "Nessuna entità trovata."

    return result, summary


ner_tab = gr.Interface(
    fn=ner_extract,
    inputs=gr.Textbox(
        label="Testo da analizzare",
        lines=6,
        placeholder="Incolla o digita il testo qui (es. trascrizione dal tab OCR Manoscritto)…",
    ),
    outputs=[
        gr.HighlightedText(
            label="Entità Nominate",
            combine_adjacent=False,
            color_map={
                "PER": "red",
                "ORG": "blue",
                "LOC": "green",
                "MISC": "orange",
            },
        ),
        gr.Markdown(label="Riepilogo entità"),
    ],
    title="Riconoscimento Entità Nominate (BERT-NER)",
    description=(
        "Estrae entità nominate da qualsiasi testo — ideale come secondo passo dopo la trascrizione OCR.\n\n"
        "**Modello:** `Babelscape/wikineural-multilingual-ner` (multilingue, supporta l'italiano)\n"
        "**Entità rilevate:** PER (persona), ORG (organizzazione), LOC (luogo), MISC (varie)\n"
        "**Uso forense:** Identificare persone, luoghi e organizzazioni citati in documenti manoscritti, "
        "lettere anonime, contratti e atti processuali."
    ),
    examples=[
        ["Mario Rossi ha firmato il contratto per conto di Acme S.r.l. a Milano il 12 marzo 2024."],
        ["Il sospettato, Maria Bianchi, è stato visto l'ultima volta vicino al Colosseo a Roma da agenti dell'Interpol."],
    ],
    flagging_mode="never",
)

# ──────────────────────────────────────────────────────────────────────────────
# Tab 5 — Graphological Feature Analysis
# ──────────────────────────────────────────────────────────────────────────────

def grapho_analyse(image: np.ndarray) -> tuple[str, np.ndarray]:
    if image is None:
        return "Carica un'immagine di scrittura a mano.", image

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
        f"**Analisi delle Caratteristiche Grafologiche**\n\n"
        f"| Caratteristica | Valore |\n"
        f"|----------------|--------|\n"
        f"| Inclinazione media lettere | {slant_mean:+.1f}° ({'destra' if slant_mean > 0 else 'sinistra' if slant_mean < 0 else 'verticale'}) |\n"
        f"| Variazione inclinazione (σ) | {slant_std:.1f}° |\n"
        f"| Pressione del tratto | {pressure_mean:.1f} / 255 |\n"
        f"| Altezza media lettere | {h_mean:.1f} px |\n"
        f"| Larghezza media lettere | {w_mean:.1f} px |\n"
        f"| Spaziatura media parole | {word_spacing:.1f} px |\n"
        f"| Densità inchiostro | {ink_density:.2f}% |\n"
        f"| Componenti connesse | {len(valid)} |\n\n"
        f"*I bounding box delle lettere sono visibili nell'immagine annotata.*"
    )
    return report, vis


grapho_tab = gr.Interface(
    fn=grapho_analyse,
    inputs=gr.Image(label="Immagine di testo manoscritto", type="numpy"),
    outputs=[
        gr.Markdown(label="Rapporto di analisi"),
        gr.Image(label="Immagine annotata (bounding box lettere)", type="numpy"),
    ],
    title="Analisi delle Caratteristiche Grafologiche",
    description=(
        "Carica un'immagine di testo manoscritto. Lo strumento estrae metriche grafologiche "
        "tra cui inclinazione delle lettere, pressione del tratto, dimensione e spaziatura.\n\n"
        "**Tecnica:** OpenCV + elaborazione classica delle immagini\n"
        "**Uso forense:** Profilazione, analisi comparativa, supporto alla perizia."
    ),
    flagging_mode="never",
)

# ──────────────────────────────────────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────────────────────────────────────

demo = gr.TabbedInterface(
    interface_list=[htr_tab, sig_verify_tab, sig_detect_tab, ner_tab, grapho_tab],
    tab_names=[
        "OCR Manoscritto",
        "Verifica Firma",
        "Rilevamento Firma",
        "Riconoscimento Entità",
        "Analisi Grafologica",
    ],
    title="GraphoLab — Intelligenza Artificiale in Grafologia Forense",
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )
