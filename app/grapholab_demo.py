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

# Prevent sklearn from probing polars (avoids circular import when polars is
# installed but not fully initialized at thread startup time)
os.environ.setdefault("SKLEARN_NO_POLARS", "1")

warnings.filterwarnings("ignore")

# Allow importing from the project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import io
import requests as _requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import gradio as gr
from collections import OrderedDict
from PIL import Image, ImageDraw, ImageFont, ImageOps
from skimage import filters, transform as sk_transform
from skimage.feature import hog, local_binary_pattern
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, pipeline as hf_pipeline
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TROCR_MODEL = "microsoft/trocr-large-handwritten"
YOLO_REPO = "tech4humans/yolov8s-signature-detector"
YOLO_FILENAME = "yolov8s.pt"
SIG_THRESHOLD = 0.35  # cosine distance threshold for signature verification
NER_MODEL = "Babelscape/wikineural-multilingual-ner"

# ──────────────────────────────────────────────────────────────────────────────
# Lazy model loaders (loaded on first use to avoid memory duplication)
# ──────────────────────────────────────────────────────────────────────────────

_trocr_processor = None
_trocr_model = None
_easyocr_reader = None
_yolo_model = None
_ner_pipeline = None
_writer_clf = None
_writer_le = None
_writer_X_scaled = None       # scaled training features for open-set distance check
_writer_dist_threshold = None  # auto-calibrated rejection threshold
import threading as _threading
import hashlib as _hashlib
_writer_lock = _threading.Lock()

# ── RAG / Ollama ──────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"
_rag_chunks: list = []   # [{"text": str, "source": str, "emb": np.ndarray}]
_rag_indexed_files: set = set()  # filenames already indexed via upload
_rag_ready = False
_rag_lock = _threading.Lock()
_RAG_CACHE_DIR = ROOT / "data" / "rag_cache"


def get_trocr():
    global _trocr_processor, _trocr_model
    if _trocr_processor is None:
        print("Loading TrOCR...")
        _trocr_processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
        _trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL).to(DEVICE)
        _trocr_model.eval()
    return _trocr_processor, _trocr_model


def get_easyocr():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        print("Loading EasyOCR (Italian)...")
        _easyocr_reader = easyocr.Reader(["it", "en"], gpu=DEVICE == "cuda")
    return _easyocr_reader


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

def _preprocess_for_htr(image: np.ndarray) -> np.ndarray:
    """Light preprocessing: deskew + contrast enhancement, keeping grayscale gradients
    so EasyOCR's CRNN recogniser retains letter-shape information."""
    import cv2

    # 1. Grayscale
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # 2. Deskew via minAreaRect on ink pixels
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(bw > 0))
    if len(coords) > 100:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        else:
            angle = -angle
        if abs(angle) > 0.3:
            (h, w) = gray.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)

    # 3. CLAHE contrast enhancement (adaptive, preserves gradients)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 4. Back to 3-channel for EasyOCR
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def htr_transcribe(image: np.ndarray) -> str:
    if image is None:
        return "Carica un'immagine di testo manoscritto."
    reader = get_easyocr()
    processed = _preprocess_for_htr(image)
    results = reader.readtext(processed, detail=0, paragraph=True)
    return "\n".join(results)


htr_tab = gr.Interface(
    fn=htr_transcribe,
    inputs=gr.Image(label="Immagine di testo manoscritto", type="numpy"),
    outputs=gr.Textbox(label="Trascrizione", lines=8),
    title="Riconoscimento Testo Manoscritto",
    description=(
        "Carica un'immagine di testo scritto a mano: il sistema lo convertirà automaticamente "
        "in testo digitale, come farebbe un dattilografo molto veloce. "
        "Funziona sia su immagini a riga singola che su documenti con più righe "
        "(le righe vengono separate automaticamente prima dell'analisi).\n\n"
        "**Quando usarlo:** lettere anonime, documenti storici, verbali scritti a mano.\n\n"
        "*Tecnologia: EasyOCR con supporto nativo italiano*"
    ),
    article=(
        "### Come leggere il risultato\n"
        "Il testo trascritto appare nella casella a destra. "
        "La qualità dipende dalla nitidezza dell'immagine: scrittura chiara su sfondo "
        "bianco dà i migliori risultati. Risoluzioni consigliate: 300 DPI o superiore.\n\n"
        "### Cosa questo strumento non fa\n"
        "Trascrive le parole così come le vede — non interpreta il significato del testo "
        "né identifica chi lo ha scritto. Può commettere errori su lettere ambigue o "
        "grafie molto personali. La trascrizione è un punto di partenza, non un prodotto finito."
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

_SIG_SAMPLES = ROOT / "data" / "samples"


def _sig_ex(name: str) -> str | None:
    p = _SIG_SAMPLES / name
    return str(p) if p.exists() else None


def sig_verify(
    ref_image: np.ndarray,
    ref_image2: np.ndarray | None,
    query_image: np.ndarray,
) -> tuple[str, np.ndarray | None]:
    if ref_image is None or query_image is None:
        return "Carica la firma di riferimento e quella da verificare.", None

    model = get_signet()

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

    # ── Matplotlib visualisation ──────────────────────────────────────────────
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

    # Gauge panel
    ax_g = axes[-1]
    ax_g.set_xlim(0, 1)
    ax_g.set_ylim(0, 1)
    ax_g.axis("off")

    # Verdict text
    ax_g.text(0.5, 0.82, verdict, ha="center", va="center",
              fontsize=14, fontweight="bold", color=color,
              transform=ax_g.transAxes)

    # Gauge bar (distance from 0 to 1)
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


_sig_examples = []
for _n in ["1", "2", "3"]:
    _r1 = _sig_ex(f"genuine_{_n}_1.png")
    _r2 = _sig_ex(f"genuine_{_n}_2.png")
    _forg = _sig_ex(f"forged_{_n}_1.png")
    if _r1 and _r2 and _forg:
        _sig_examples.append([_r1, _r2, _forg])       # FALSA con 2 refs
    if _n == "1" and _r1 and _r2:
        _sig_examples.append([_r1, None, _r2])         # AUTENTICA con 1 ref


sig_verify_tab = gr.Interface(
    fn=sig_verify,
    inputs=[
        gr.Image(label="Firma di riferimento 1 (autentica nota)", type="numpy"),
        gr.Image(label="Firma di riferimento 2 — opzionale (migliora l'accuratezza)", type="numpy"),
        gr.Image(label="Firma da verificare", type="numpy"),
    ],
    outputs=[
        gr.Textbox(label="Risultato della verifica", lines=8),
        gr.Image(label="Confronto visivo", type="numpy"),
    ],
    title="Verifica Autenticità Firma",
    description=(
        "Confronta una firma autentica nota con una firma da esaminare. "
        "Il sistema misura quanto le due firme si assomigliano nello stile visivo "
        "e produce un giudizio accompagnato da un grafico.\n\n"
        "Puoi caricare fino a **due firme di riferimento**: usarne due riduce il rischio "
        "di errore dovuto alla naturale variabilità della stessa firma nel tempo.\n\n"
        "**Quando usarlo:** assegni bancari contestati, contratti, testamenti, documenti d'identità.\n\n"
        "*Tecnologia: rete neurale specializzata nel confronto di firme (SigNet, addestrata su migliaia di campioni)*"
    ),
    article=(
        "### Come leggere il risultato\n"
        "- **AUTENTICA ✓** — le caratteristiche visive della firma esaminata corrispondono ai riferimenti. "
        "La barra nel grafico è corta (le firme sono simili).\n"
        "- **FALSA ✗** — le caratteristiche visive differiscono in modo significativo. "
        "La barra è lunga (le firme sono diverse).\n"
        "- La **linea tratteggiata** nel grafico indica la soglia di decisione: "
        "a sinistra = autentica, a destra = falsa.\n\n"
        "### Cosa questo strumento non fa\n"
        "Non emette un verdetto legale definitivo: fornisce un'indicazione quantitativa "
        "che il perito valuta insieme ad altri elementi. "
        "Una firma autentica di uno scrittore anziano o in condizioni di salute diverse "
        "può risultare 'diversa' da quella giovanile. "
        "Il giudizio finale spetta sempre al perito calligrafo qualificato."
    ),
    examples=_sig_examples if _sig_examples else None,
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
                  label="Soglia di confidenza (valori bassi = più sensibile, valori alti = più selettivo)"),
    ],
    outputs=[
        gr.Image(label="Documento annotato con le firme rilevate", type="numpy"),
        gr.Markdown(label="Riepilogo rilevamento"),
    ],
    title="Rilevamento Firme nei Documenti",
    description=(
        "Carica l'immagine di un documento scansionato: il sistema individuerà automaticamente "
        "tutte le firme presenti e le evidenzierà con un riquadro colorato.\n\n"
        "La **soglia di confidenza** regola la sensibilità del rilevamento: "
        "valori bassi trovano più firme (ma con qualche falso positivo); "
        "valori alti trovano solo le firme più chiare e definite.\n\n"
        "**Quando usarlo:** contratti multipagina, atti notarili, moduli bancari, assegni, "
        "qualsiasi documento in cui occorra localizzare rapidamente le firme.\n\n"
        "*Tecnologia: rete di rilevamento oggetti addestrata specificamente su firme manoscritte*"
    ),
    article=(
        "### Come leggere il risultato\n"
        "Ogni riquadro blu sull'immagine indica una firma rilevata, con accanto la percentuale "
        "di fiducia del sistema (es. 'Sig #1  87%'). "
        "Le firme rilevate possono essere estratte e passate al tab **Verifica Firma** "
        "per un'analisi di autenticità.\n\n"
        "### Cosa questo strumento non fa\n"
        "Individua la *posizione* delle firme nel documento, ma non ne valuta l'autenticità. "
        "Elementi grafici simili a una firma — timbri, decorazioni, iniziali — possono "
        "occasionalmente essere segnalati erroneamente. "
        "Questo tab è il primo passo di una pipeline: rileva → estrai → verifica."
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
        placeholder="Incolla o digita il testo qui (es. trascrizione prodotta dal tab OCR Manoscritto)…",
    ),
    outputs=[
        gr.HighlightedText(
            label="Testo con entità evidenziate",
            combine_adjacent=False,
            color_map={
                "PER": "red",
                "ORG": "blue",
                "LOC": "green",
                "MISC": "orange",
            },
        ),
        gr.Markdown(label="Elenco entità trovate"),
    ],
    title="Riconoscimento di Persone, Luoghi e Organizzazioni",
    description=(
        "Incolla un testo — ad esempio la trascrizione prodotta dal tab **OCR Manoscritto** — "
        "e il sistema identificherà automaticamente tutte le persone, i luoghi e le "
        "organizzazioni menzionati, evidenziandoli con colori diversi.\n\n"
        "**Quando usarlo:** lettere anonime trascritte, dichiarazioni giurate, atti processuali — "
        "ovunque occorra estrarre rapidamente i soggetti coinvolti senza leggere l'intero documento.\n\n"
        "*Tecnologia: modello linguistico multilingue addestrato su testi in italiano, inglese, "
        "tedesco, spagnolo e altre lingue*"
    ),
    article=(
        "### Come leggere il risultato\n"
        "Il testo viene evidenziato con colori diversi in base al tipo di entità trovata:\n\n"
        "🔴 **Rosso = Persona** &nbsp;|&nbsp; "
        "🔵 **Blu = Organizzazione** &nbsp;|&nbsp; "
        "🟢 **Verde = Luogo** &nbsp;|&nbsp; "
        "🟠 **Arancione = Altra entità rilevante**\n\n"
        "La tabella sotto il testo riporta ogni entità trovata con la percentuale di fiducia "
        "del sistema. Valori superiori all'80% indicano un riconoscimento affidabile.\n\n"
        "### Cosa questo strumento non fa\n"
        "Identifica le entità in base alla loro forma linguistica, non alla loro "
        "rilevanza giuridica. Nomi comuni che coincidono con nomi propri possono "
        "essere riconosciuti erroneamente. "
        "Decidere quali entità siano pertinenti al caso spetta sempre all'investigatore."
    ),
    examples=[
        ["Mario Rossi ha firmato il contratto per conto di Acme S.r.l. a Milano il 12 marzo 2024."],
        ["Il sospettato, Maria Bianchi, è stato visto l'ultima volta vicino al Colosseo a Roma da agenti dell'Interpol."],
    ],
    flagging_mode="never",
)

# ──────────────────────────────────────────────────────────────────────────────
# Tab 5 — Writer Identification
# ──────────────────────────────────────────────────────────────────────────────

WRITER_IMG_SIZE = (128, 256)   # (H, W) for feature extraction
_WRITER_SAMPLES_DIR = ROOT / "data" / "samples"
_WRITER_EXAMPLES_DIR = ROOT / "data" / "samples" / "writer_examples"

_WRITER_NAMES = {
    0: "Scrittore A",
    1: "Scrittore B",
    2: "Scrittore C",
    3: "Scrittore D",
    4: "Scrittore E",
}


def _make_synthetic_writer(writer_id: int, sample_id: int) -> Image.Image:
    """Generate a synthetic handwriting sample using system TTF fonts."""
    rng = np.random.default_rng(writer_id * 1000 + sample_id)

    _FONTS_DIR = Path("C:/Windows/Fonts")
    # Each writer gets a distinct handwriting font + base size
    _WRITER_FONTS = [
        ("Inkfree.ttf",  19),   # Writer 0 — Ink Free (corsivo informale)
        ("LHANDW.TTF",   17),   # Writer 1 — Lucida Handwriting (elegante)
        ("segoepr.ttf",  18),   # Writer 2 — Segoe Print (stampatello mano)
        ("segoesc.ttf",  16),   # Writer 3 — Segoe Script (corsivo moderno)
        ("comic.ttf",    18),   # Writer 4 — Comic Sans (tondo informale)
    ]
    font_name, base_size = _WRITER_FONTS[writer_id % len(_WRITER_FONTS)]
    font_size = base_size + int(rng.integers(-1, 2))
    try:
        font = ImageFont.truetype(str(_FONTS_DIR / font_name), font_size)
    except Exception:
        font = ImageFont.load_default()

    # Ink darkness: each writer has a characteristic pen pressure
    ink_value = int([25, 15, 35, 20, 30][writer_id % 5] + rng.integers(-5, 6))

    _SENTENCES = [
        "il gatto dorme sul tetto",
        "la casa è piccola e bella",
        "oggi il cielo è molto blu",
        "scrivere a mano è un'arte",
        "ogni persona ha uno stile",
        "il sole tramonta a ovest",
        "leggo un libro ogni sera",
        "la penna scorre sul foglio",
        "le parole raccontano storie",
        "questo è un campione scritto",
    ]
    lines = [
        _SENTENCES[(writer_id * 3 + sample_id + i) % len(_SENTENCES)]
        for i in range(3)
    ]

    w, h = 320, 140
    img = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(img)

    line_gap = font_size + 12 + int(rng.integers(-2, 3))
    y = 10
    for line in lines:
        x = 8 + int(rng.integers(-3, 4))
        draw.text((x, y), line, fill=ink_value, font=font)
        y += line_gap

    # Slight rotation simulates unaligned paper
    angle = float(rng.uniform(-1.5, 1.5))
    img = img.rotate(angle, fillcolor=255, expand=False)

    return img


def _preprocess_writer_img(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL image to normalised grayscale array of WRITER_IMG_SIZE.

    For portrait documents (full page), extracts a representative landscape
    crop from the text body before resizing, preserving stroke-level features
    that the model was trained on (word-level 320×140 samples).
    """
    gray = pil_img.convert("L")
    w, h = gray.size
    # If image is portrait, take a landscape crop from the upper text body
    # (avoids distorting full-page documents when resizing to landscape target)
    target_ratio = WRITER_IMG_SIZE[1] / WRITER_IMG_SIZE[0]  # 256/128 = 2.0
    if h > w:
        crop_h = int(w / target_ratio)
        top = h // 6  # skip top margin, start from first text lines
        top = min(top, max(0, h - crop_h))
        gray = gray.crop((0, top, w, top + crop_h))
    arr = np.array(gray, dtype=np.float32)
    # OTSU threshold → invert so ink=1, bg=0
    thresh = filters.threshold_otsu(arr) if arr.std() > 1 else 128.0
    binary = (arr < thresh).astype(np.float32)
    # Resize
    resized = sk_transform.resize(binary, WRITER_IMG_SIZE, anti_aliasing=True)
    return resized.astype(np.float32)


def _extract_writer_features(pil_img: Image.Image) -> np.ndarray:
    """Extract HOG + LBP + run-length features for writer identification."""
    arr = _preprocess_writer_img(pil_img)
    arr8 = (arr * 255).astype(np.uint8)

    # HOG features
    hog_feats = hog(
        arr,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True,
    )

    # LBP histogram (26 uniform bins)
    lbp = local_binary_pattern(arr8, P=24, R=3, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=26, range=(0, 26), density=True)

    # Run-length statistics (horizontal & vertical)
    def _run_stats(binary_row):
        runs = []
        cnt = 0
        for v in binary_row:
            if v > 0.5:
                cnt += 1
            elif cnt > 0:
                runs.append(cnt)
                cnt = 0
        if cnt > 0:
            runs.append(cnt)
        return runs

    h_runs = []
    for row in arr:
        h_runs.extend(_run_stats(row))
    v_runs = []
    for col in arr.T:
        v_runs.extend(_run_stats(col))

    h_arr = np.array(h_runs, dtype=np.float32) if h_runs else np.array([0.0])
    v_arr = np.array(v_runs, dtype=np.float32) if v_runs else np.array([0.0])
    run_feats = np.array([
        h_arr.mean(), h_arr.std(), h_arr.max(),
        v_arr.mean(), v_arr.std(), v_arr.max(),
    ], dtype=np.float32)

    return np.concatenate([hog_feats, lbp_hist, run_feats])


def _load_real_writer_samples() -> tuple[list[np.ndarray], list[str]] | None:
    """Load samples from data/samples/writer_XX/sample_YY.png directories."""
    writer_dirs = sorted(_WRITER_SAMPLES_DIR.glob("writer_??"))
    if len(writer_dirs) < 2:
        return None
    X, y = [], []
    for wd in writer_dirs:
        samples = sorted(wd.glob("sample_*.png"))
        if len(samples) < 3:
            continue
        for sp in samples:
            try:
                img = Image.open(sp)
                X.append(_extract_writer_features(img))
                y.append(wd.name)
            except Exception:
                pass
    if len(set(y)) < 2:
        return None
    return X, y


def _get_writer_model():
    """Return (Pipeline, LabelEncoder), training lazily on first call.

    Thread-safe: if the background pre-warm thread is still running when the
    pipeline reaches step 4, this call blocks until training finishes rather
    than spawning a duplicate training job.
    """
    global _writer_clf, _writer_le
    if _writer_clf is not None:          # fast path — no lock needed
        return _writer_clf, _writer_le
    with _writer_lock:                   # only one thread trains at a time
        if _writer_clf is not None:      # re-check after acquiring lock
            return _writer_clf, _writer_le
        print("Training writer identification model...")

    real = _load_real_writer_samples()
    if real is not None:
        X_raw, y_raw = real
        labels = y_raw
    else:
        # Synthetic fallback: 5 writers × 10 samples
        X_raw, labels = [], []
        for wid in range(5):
            for sid in range(10):
                img = _make_synthetic_writer(wid, sid)
                X_raw.append(_extract_writer_features(img))
                labels.append(_WRITER_NAMES[wid])

    le = LabelEncoder()
    y_enc = le.fit_transform(labels)
    X = np.array(X_raw)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=10, gamma="scale", probability=True)),
    ])
    clf.fit(X, y_enc)

    # Open-set calibration: compute max intra-class nearest-neighbour distance
    # in the scaled feature space, then use 2× as the rejection threshold.
    global _writer_X_scaled, _writer_dist_threshold
    X_scaled = clf.named_steps["scaler"].transform(X)
    max_intra = 0.0
    for cls in np.unique(y_enc):
        Xc = X_scaled[y_enc == cls]
        if len(Xc) < 2:
            continue
        diff = Xc[:, np.newaxis, :] - Xc[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        max_intra = max(max_intra, dists.min(axis=1).max())
    _writer_X_scaled = X_scaled
    _writer_dist_threshold = max_intra * 2.0

    _writer_clf = clf
    _writer_le = le
    print(f"Writer model ready — {len(le.classes_)} writers, {len(X)} samples. "
          f"Rejection threshold: {_writer_dist_threshold:.3f}")
    return _writer_clf, _writer_le


def _ensure_writer_examples() -> list[str]:
    """Pre-generate example images for the Gradio examples list."""
    _WRITER_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for wid in range(5):
        p = _WRITER_EXAMPLES_DIR / f"writer_{wid}_example.png"
        if not p.exists():
            img = _make_synthetic_writer(wid, sample_id=99)
            img.save(str(p))
        paths.append(str(p))
    return paths


_writer_example_paths = _ensure_writer_examples()

# Pre-warm writer model in background so step 4 of pipeline is instant
_threading.Thread(target=_get_writer_model, daemon=True).start()


def writer_identify(image: np.ndarray) -> tuple[str, np.ndarray]:
    if image is None:
        return "Carica un'immagine di testo manoscritto.", None
    try:
        clf, le = _get_writer_model()
    except Exception as e:
        return f"Errore nel caricamento del modello: {e}", None

    pil_img = Image.fromarray(image)
    try:
        feat = _extract_writer_features(pil_img)
    except Exception as e:
        return f"Errore nell'estrazione delle caratteristiche: {e}", None

    proba = clf.predict_proba([feat])[0]
    order = np.argsort(proba)[::-1]
    names = le.inverse_transform(order)
    scores = proba[order]

    # Open-set check: nearest-neighbour distance in scaled feature space
    is_unknown = False
    if _writer_X_scaled is not None and _writer_dist_threshold is not None:
        feat_scaled = clf.named_steps["scaler"].transform([feat])[0]
        min_dist = np.linalg.norm(_writer_X_scaled - feat_scaled, axis=1).min()
        is_unknown = min_dist > _writer_dist_threshold

    # Markdown report
    rows = "\n".join(
        f"| {'🥇' if i == 0 else '🥈' if i == 1 else '🥉' if i == 2 else '  '} "
        f"**{name}** | {score:.1%} |"
        for i, (name, score) in enumerate(zip(names, scores))
    )
    if is_unknown:
        report = (
            "**⚠️ Scrittore non identificato nel database**\n\n"
            "La scrittura analizzata non corrisponde a nessuno degli scrittori noti. "
            "Le probabilità di seguito hanno valore puramente indicativo "
            "e **non devono essere usate per un'attribuzione**.\n\n"
            "| Candidato | Probabilità (riferimento) |\n"
            "|-----------|---------------------------|\n"
            + rows
            + "\n\n*La distanza dal campione più simile nel database supera la soglia "
              "di affidabilità. Aggiungere campioni dello scrittore al database per "
              "un confronto diretto.*"
        )
    else:
        report = (
            "**Identificazione Scrittore — Risultati**\n\n"
            "| Candidato | Probabilità |\n"
            "|-----------|-------------|\n"
            + rows
            + "\n\n*I risultati si basano su caratteristiche HOG + LBP + statistiche dei tratti.*"
        )
    if _load_real_writer_samples() is None:
        report += (
            "\n\n⚠️ *Dati sintetici: il modello è addestrato su scritture generate "
            "artificialmente. Per risultati forensi reali, popola `data/samples/writer_XX/`.*"
        )

    # Bar chart
    fig, ax = plt.subplots(figsize=(5, max(2.5, len(names) * 0.55)))
    if is_unknown:
        colors = ["#aaaaaa"] * len(names)
        chart_title = "Scrittore non nel database — solo riferimento"
    else:
        colors = ["#1B3A6B" if i == 0 else "#C8973A" if i == 1 else "#9eb8e0"
                  for i in range(len(names))]
        chart_title = "Probabilità per scrittore"
    ax.barh(names[::-1], scores[::-1] * 100, color=colors[::-1])
    ax.set_xlabel("Probabilità (%)")
    ax.set_xlim(0, 105)
    ax.set_title(chart_title)
    for i, (name, score) in enumerate(zip(names[::-1], scores[::-1])):
        ax.text(score * 100 + 1, i, f"{score:.1%}", va="center", fontsize=9)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    chart_arr = np.array(Image.open(buf))

    return report, chart_arr


writer_tab = gr.Interface(
    fn=writer_identify,
    inputs=gr.Image(label="Campione di scrittura a mano da attribuire", type="numpy"),
    outputs=[
        gr.Markdown(label="Candidati ordinati per probabilità"),
        gr.Image(label="Grafico delle probabilità", type="numpy"),
    ],
    title="Identificazione dello Scrittore",
    description=(
        "Carica un campione di scrittura a mano: il sistema estrarrà automaticamente "
        "le caratteristiche grafologiche — forma delle lettere, texture del tratto, "
        "ritmo della spaziatura — e confronterà lo stile con quello degli scrittori "
        "nel database, producendo una lista di candidati ordinata per probabilità.\n\n"
        "**Quando usarlo:** attribuzione di autoria in lettere anonime, note manoscritte, "
        "documenti contestati tra più parti.\n\n"
        "*Tecnologia: analisi automatica delle caratteristiche grafologiche + classificatore statistico*"
    ),
    article=(
        "### Come leggere il risultato\n"
        "Il grafico a barre mostra la probabilità che la scrittura appartenga a ciascuno "
        "scrittore nel database. Il candidato con la barra più lunga è quello il cui stile "
        "grafico è più simile al campione caricato.\n\n"
        "⚠️ **Nota sulla demo:** in questa versione dimostrativa il sistema è addestrato "
        "su campioni sintetici (scritture generate artificialmente con stili diversi). "
        "Per un uso forense reale occorre addestrare il modello su campioni autentici "
        "degli scrittori candidati, organizzati nella cartella `data/samples/writer_XX/`.\n\n"
        "### Cosa questo strumento non fa\n"
        "Anche una probabilità elevata non costituisce prova dell'autoria: è un'indicazione "
        "statistica che suggerisce su quali soggetti concentrare l'esame peritale. "
        "Il risultato va sempre valutato da un perito calligrafo qualificato insieme "
        "ad altri elementi probatori."
    ),
    examples=[[p] for p in _writer_example_paths],
    flagging_mode="never",
)

# ──────────────────────────────────────────────────────────────────────────────
# Tab 6 — Graphological Feature Analysis
# ──────────────────────────────────────────────────────────────────────────────

def grapho_analyse(image: np.ndarray) -> tuple[str, np.ndarray]:
    if image is None:
        return "Carica un'immagine di scrittura a mano.", image

    # Cap to 800 px: adaptive threshold is O(pixels × blockSize), so keeping
    # the image small is critical. Graphological metrics are scale-invariant.
    h0, w0 = image.shape[:2]
    if max(h0, w0) > 800:
        sc = 800 / max(h0, w0)
        image = cv2.resize(image, (int(w0 * sc), int(h0 * sc)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    # Adaptive threshold works locally: ignores the global dark background of
    # phone photos that fools global Otsu into treating borders as ink.
    binary = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (5, 5), 0), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10,
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
        gr.Markdown(label="Scheda delle caratteristiche grafologiche"),
        gr.Image(label="Immagine annotata (ogni lettera evidenziata)", type="numpy"),
    ],
    title="Analisi delle Caratteristiche Grafologiche",
    description=(
        "Carica un'immagine di testo manoscritto: il sistema misurerà automaticamente "
        "le principali caratteristiche grafologiche — inclinazione delle lettere, "
        "pressione del tratto, dimensioni e spaziatura — producendo una scheda metrica oggettiva.\n\n"
        "**Quando usarlo:** analisi comparativa tra due campioni dello stesso documento, "
        "verifica della coerenza interna di un testo, supporto alla perizia calligrafica.\n\n"
        "*Tecnologia: elaborazione digitale dell'immagine con rilevamento automatico delle lettere*"
    ),
    article=(
        "### Come leggere i valori\n\n"
        "| Caratteristica | Significato forense |\n"
        "|---|---|\n"
        "| **Inclinazione** | Tende ad essere costante nei campioni autentici dello stesso scrittore; "
        "variazioni anomale possono segnalare un tentativo di camuffamento |\n"
        "| **Pressione del tratto** | Dipende dalla penna e dallo stato emotivo; "
        "differenze marcate tra sezioni dello stesso documento meritano attenzione |\n"
        "| **Altezza/Larghezza lettere** | Valori molto diversi tra campioni diversi "
        "possono suggerire scrittori diversi |\n"
        "| **Spaziatura parole** | Irregolarità possono indicare incertezza, "
        "interruzioni o alterazione del testo |\n\n"
        "### Cosa questo strumento non fa\n"
        "Produce misurazioni numeriche oggettive, ma non formula giudizi forensi autonomi. "
        "L'interpretazione dei valori in chiave peritale — e la loro rilevanza nel caso specifico — "
        "spetta al perito calligrafo qualificato."
    ),
    flagging_mode="never",
)

# ──────────────────────────────────────────────────────────────────────────────
# Tab 7 — Forensic Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _detect_and_crop(
    image: np.ndarray,
    conf_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Run YOLO signature detection and return (annotated, first_crop, summary).

    Gracefully degrades when YOLO is not available (missing HF_TOKEN).
    """
    annotated = image.copy()
    try:
        yolo = get_yolo()
    except Exception:
        return annotated, None, "⚠️ Rilevamento firma non disponibile (HF_TOKEN mancante)."

    pil_img = Image.fromarray(image).convert("RGB")
    import tempfile
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


def run_pipeline(
    doc_image: np.ndarray,
    ref_sig: np.ndarray | None,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    """Orchestrate all 6 AI tools in sequence.

    Generator: yields partial results after each step so the UI updates live.
    Output order: s1_img, s1_txt, s2_txt, s3_hl, s3_md,
                  s4_md, s4_img, s5_md, s5_img,
                  s6_txt, s6_img, final_md, pipe_results
    """
    _ = gr.update()   # no-op: leave output unchanged

    if doc_image is None:
        msg = "Carica il documento da analizzare."
        yield (doc_image, msg, msg, [], msg, msg, None, msg, doc_image, msg, None, msg,
               gr.update(visible=False))
        return

    # ── Step 1: Signature Detection ───────────────────────────────────────────
    progress(0.05, desc="Step 1/6 — Rilevamento firma…")
    step1_img, sig_crop, step1_summary = _detect_and_crop(doc_image)
    yield (step1_img, step1_summary, _, _, _, _, _, _, _, _, _, _, gr.update(visible=True))

    # ── Step 2: HTR ───────────────────────────────────────────────────────────
    progress(0.20, desc="Step 2/6 — Trascrizione HTR…")
    step2_text = htr_transcribe(doc_image)
    yield (_, _, step2_text, _, _, _, _, _, _, _, _, _, _)

    # ── Step 3: NER ───────────────────────────────────────────────────────────
    progress(0.45, desc="Step 3/6 — Riconoscimento entità…")
    text_for_ner = step2_text if step2_text and step2_text.strip() else ""
    if text_for_ner:
        step3_hl, step3_summary = ner_extract(text_for_ner)
    else:
        step3_hl, step3_summary = [], "Nessun testo trascritto disponibile per il NER."
    yield (_, _, _, step3_hl, step3_summary, _, _, _, _, _, _, _, _)

    # ── Step 4: Writer Identification ─────────────────────────────────────────
    progress(0.60, desc="Step 4/6 — Identificazione scrittore…")
    step4_report, step4_chart = writer_identify(doc_image)
    yield (_, _, _, _, _, step4_report, step4_chart, _, _, _, _, _, _)

    # ── Step 5: Graphological Analysis ────────────────────────────────────────
    progress(0.75, desc="Step 5/6 — Analisi grafologica…")
    step5_report, step5_vis = grapho_analyse(doc_image)
    yield (_, _, _, _, _, _, _, step5_report, step5_vis, _, _, _, _)

    # ── Step 6: Signature Verification ────────────────────────────────────────
    progress(0.88, desc="Step 6/6 — Verifica firma…")
    if ref_sig is not None:
        query_for_verify = sig_crop if sig_crop is not None else doc_image
        step6_report, step6_chart = sig_verify(ref_sig, None, query_for_verify)
        if sig_crop is None:
            step6_report += "\n\n⚠️ Nessuna firma estratta — confronto eseguito sull'immagine intera."
    else:
        step6_report = (
            "Firma di riferimento non fornita.\n\n"
            "Per abilitare questo step carica una firma autentica nota "
            "nel campo 'Firma di riferimento' sopra."
        )
        step6_chart = None
    yield (_, _, _, _, _, _, _, _, _, step6_report, step6_chart, _, _)

    # ── Referto finale ────────────────────────────────────────────────────────
    final_report = (
        "## Referto Forense Integrato\n\n"
        "---\n\n"
        f"### Step 1 — Rilevamento Firma\n{step1_summary}\n\n"
        f"### Step 2 — Trascrizione HTR\n```\n{step2_text}\n```\n\n"
        f"### Step 3 — Entità Nominate\n{step3_summary}\n\n"
        f"### Step 4 — Identificazione Scrittore\n{step4_report}\n\n"
        f"### Step 5 — Caratteristiche Grafologiche\n{step5_report}\n\n"
        f"### Step 6 — Verifica Firma\n{step6_report}\n\n"
        "---\n\n"
        "*Referto generato automaticamente da GraphoLab. "
        "Tutti i risultati hanno carattere indicativo e devono essere valutati "
        "da un perito calligrafo qualificato.*"
    )
    yield (_, _, _, _, _, _, _, _, _, _, _, final_report, _)


with gr.Blocks() as pipeline_tab:
    gr.Markdown(
        "## Pipeline Forense Integrata\n\n"
        "Carica il documento da esaminare (es. testamento olografo, lettera anonima, contratto) "
        "e, opzionalmente, una firma di riferimento autentica. "
        "Il sistema eseguirà in sequenza tutti e sei gli strumenti AI e produrrà un **referto forense integrato**.\n\n"
        "| Step | Strumento | Input |\n"
        "|------|-----------|-------|\n"
        "| 1 | Rilevamento Firma (YOLOv8) | Documento |\n"
        "| 2 | Trascrizione HTR (EasyOCR) | Documento |\n"
        "| 3 | Riconoscimento Entità — NER | Testo da Step 2 |\n"
        "| 4 | Identificazione Scrittore | Documento |\n"
        "| 5 | Analisi Grafologica | Documento |\n"
        "| 6 | Verifica Firma (SigNet) | Firma rif. + crop da Step 1 |\n"
    )

    with gr.Row():
        pipe_doc = gr.Image(
            label="Documento da analizzare (testamento, lettera, atto)",
            type="numpy",
        )
        pipe_ref = gr.Image(
            label="Firma di riferimento nota — opzionale (per Step 6)",
            type="numpy",
        )

    pipe_btn = gr.Button("▶  Avvia Analisi Forense", variant="primary", size="lg")

    with gr.Column(visible=False) as pipe_results:
        gr.Markdown("### Step 1 — Rilevamento Firma (YOLOv8)")
        with gr.Row():
            out_s1_img = gr.Image(label="Documento annotato", type="numpy")
            out_s1_txt = gr.Textbox(label="Riepilogo", lines=3)

        gr.Markdown("### Step 2 — Trascrizione HTR (EasyOCR)")
        out_s2_txt = gr.Textbox(label="Testo trascritto", lines=6)

        gr.Markdown("### Step 3 — Riconoscimento Entità (NER)")
        out_s3_hl = gr.HighlightedText(
            label="Testo con entità evidenziate",
            combine_adjacent=False,
            color_map={"PER": "red", "ORG": "blue", "LOC": "green", "MISC": "orange"},
        )
        out_s3_md = gr.Markdown()

        gr.Markdown("### Step 4 — Identificazione Scrittore")
        with gr.Row():
            out_s4_md = gr.Markdown()
            out_s4_img = gr.Image(label="Probabilità per scrittore", type="numpy")

        gr.Markdown("### Step 5 — Analisi Grafologica")
        with gr.Row():
            out_s5_md = gr.Markdown()
            out_s5_img = gr.Image(label="Immagine annotata", type="numpy")

        gr.Markdown("### Step 6 — Verifica Firma (SigNet)")
        with gr.Row():
            out_s6_txt = gr.Textbox(label="Esito verifica", lines=6)
            out_s6_img = gr.Image(label="Confronto visivo", type="numpy")

        gr.Markdown("---")
        out_final = gr.Markdown()

    pipe_btn.click(
        fn=run_pipeline,
        inputs=[pipe_doc, pipe_ref],
        outputs=[
            out_s1_img, out_s1_txt,
            out_s2_txt,
            out_s3_hl, out_s3_md,
            out_s4_md, out_s4_img,
            out_s5_md, out_s5_img,
            out_s6_txt, out_s6_img,
            out_final,
            pipe_results,
        ],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tab 8 — Datazione Documenti
# ──────────────────────────────────────────────────────────────────────────────

import re as _re
from datetime import datetime as _datetime

# Regex L1 — date italiane e numeriche
_DATE_PATTERNS = [
    # "10 gennaio 2024" / "10 gennaio del 2024"
    r"\b(\d{1,2})\s+(gennaio|febbraio|marzo|aprile|maggio|giugno|"
    r"luglio|agosto|settembre|ottobre|novembre|dicembre)\s+(?:del\s+)?(\d{4})\b",
    # "10 gen. 2024" / abbreviazioni
    r"\b(\d{1,2})\s+(gen|feb|mar|apr|mag|giu|lug|ago|set|ott|nov|dic)\.?\s+(\d{4})\b",
    # "10/01/2024" o "10-01-2024" o "10.01.2024"
    r"\b(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})\b",
    # "gennaio 2024" (senza giorno)
    r"\b(gennaio|febbraio|marzo|aprile|maggio|giugno|"
    r"luglio|agosto|settembre|ottobre|novembre|dicembre)\s+(\d{4})\b",
]
_DATE_RE = _re.compile("|".join(_DATE_PATTERNS), _re.IGNORECASE)


def _try_dateparser(raw: str) -> _datetime | None:
    """Parse a raw date string to datetime using dateparser (Italian-aware)."""
    try:
        import dateparser
        dt = dateparser.parse(
            raw,
            languages=["it", "en"],
            settings={"PREFER_DAY_OF_MONTH": "first", "RETURN_AS_TIMEZONE_AWARE": False},
        )
        if dt and 1800 < dt.year < 2200:
            return dt
    except Exception:
        pass
    return None


def extract_dates(text: str) -> list[tuple[str, _datetime]]:
    """Extract and normalize dates from OCR text.

    Returns a list of (raw_string, datetime) pairs, sorted chronologically.
    Uses regex L1 first; falls back to scanning NER DATE entities if nothing found.
    """
    found: list[tuple[str, _datetime]] = []

    # L1 — regex
    _BIRTH_KW = ("nata", "nato", "nascita", "nasc.", "nata il", "nato il")
    for m in _DATE_RE.finditer(text):
        raw = m.group(0).strip()
        context_before = text[max(0, m.start() - 35) : m.start()].lower()
        if any(kw in context_before for kw in _BIRTH_KW):
            continue  # data di nascita — ignorala
        dt = _try_dateparser(raw)
        if dt:
            found.append((raw, dt))

    # L2 — NER fallback (filters DATE entities from wikineural NER)
    if not found:
        try:
            ner_html, ner_md = ner_extract(text)
            # Extract DATE spans from NER Markdown (pattern: **WORD** `DATE`)
            for raw in _re.findall(r"\*\*([^*]+)\*\*\s*`DATE`", ner_md or ""):
                dt = _try_dateparser(raw)
                if dt:
                    found.append((raw, dt))
        except Exception:
            pass

    # De-duplicate by normalized date
    seen: set[str] = set()
    unique: list[tuple[str, _datetime]] = []
    for raw, dt in found:
        key = dt.strftime("%Y-%m-%d")
        if key not in seen:
            seen.add(key)
            unique.append((raw, dt))

    return sorted(unique, key=lambda x: x[1])


def dating_rank(files: list) -> str:
    """Main function for the Datazione Documenti tab.

    Accepts a list of uploaded files (gr.File objects), runs OCR on each,
    extracts dates, and returns a Markdown table sorted chronologically.
    """
    if not files:
        return "Carica almeno un'immagine di documento."

    reader = get_easyocr()
    rows: list[tuple[str, str, _datetime | None]] = []

    for f in files:
        path = f.name if hasattr(f, "name") else str(f)
        name = path.split("\\")[-1].split("/")[-1]
        try:
            img = Image.open(path).convert("RGB")
            img_np = np.array(img)
            ocr_lines = reader.readtext(img_np, detail=0, paragraph=False)
            text = "\n".join(ocr_lines)
            dates = extract_dates(text)
            if dates:
                raw, dt = dates[-1]         # data più recente = data di redazione
                rows.append((name, raw, dt))
            else:
                rows.append((name, "—  data non trovata", None))
        except Exception as e:
            rows.append((name, f"Errore: {e}", None))

    # Sort: dated docs first (chronologically), undated last
    dated   = [(n, r, dt) for n, r, dt in rows if dt is not None]
    undated = [(n, r, dt) for n, r, dt in rows if dt is None]
    dated.sort(key=lambda x: x[2])
    sorted_rows = dated + undated

    lines = [
        "## Datazione Documenti — Risultati\n",
        "| # | Documento | Data estratta | Data normalizzata |",
        "|---|-----------|--------------|-------------------|",
    ]
    for i, (name, raw, dt) in enumerate(sorted_rows, 1):
        norm = dt.strftime("%Y-%m-%d") if dt else "—"
        lines.append(f"| {i} | `{name}` | {raw} | {norm} |")

    if not dated:
        lines.append("\n> Nessuna data rilevata nei documenti caricati.")
    else:
        lines.append(
            f"\n*{len(dated)} document{'o' if len(dated)==1 else 'i'} datato/i, "
            f"{len(undated)} senza data.*"
        )

    return "\n".join(lines)


dating_tab = gr.Interface(
    fn=dating_rank,
    inputs=gr.File(
        label="Immagini documenti (carica 2 o più)",
        file_count="multiple",
        file_types=["image"],
    ),
    outputs=gr.Markdown(label="Documenti ordinati per data"),
    title="Datazione Documenti",
    description=(
        "Carica più immagini di documenti manoscritti o stampati: il sistema estrarrà "
        "le date presenti nel testo e restituirà i documenti ordinati cronologicamente.\n\n"
        "**Quando usarlo:** confrontare testamenti di date diverse, ordinare una corrispondenza, "
        "ricostruire la sequenza temporale di un caso.\n\n"
        "*Tecnologia: EasyOCR + regex italiana + dateparser multilingue*"
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 9 — Consulente Forense IA (RAG + Ollama)
# ──────────────────────────────────────────────────────────────────────────────

_RAG_SYNTHETIC_DOCS = [
    (
        "Analisi della pressione",
        "La pressione grafica indica la forza con cui la penna o la matita viene premuta sul foglio. "
        "Una pressione forte (tratti profondi, rilevabili anche sul retro del foglio) è associata a "
        "carattere deciso, vitalità e a volte aggressività. Una pressione leggera (tratti quasi "
        "impercettibili) può indicare sensibilità, adattabilità o, in contesti patologici, stanchezza "
        "e astenia. La pressione irregolare — alternanza di tratti forti e deboli nello stesso scritto — "
        "può segnalare instabilità emotiva, stati di ansia o condizioni neurologiche. In grafologia "
        "forense la pressione è fondamentale per distinguere scritture apposte in condizioni normali "
        "da quelle prodotte sotto costrizione fisica o psicologica.",
    ),
    (
        "Inclinazione del tratto",
        "L'inclinazione della scrittura descrive l'angolo dei tratti verticali delle lettere rispetto "
        "alla riga di base. Una scrittura verticale (0°) indica equilibrio e obiettività. "
        "L'inclinazione a destra (>15°) è associata a estroversia, impulsività e orientamento verso "
        "il futuro. L'inclinazione a sinistra (<−10°) può indicare introversione, tendenza al ripiegamento "
        "su se stessi o, in contesti forensi, un tentativo di camuffare la propria calligrafia. "
        "L'inclinazione variabile (misto destra/sinistra nello stesso testo) è indicatore di "
        "instabilità emotiva. La misurazione forense dell'inclinazione avviene tramite analisi "
        "angolare dei tratti ascendenti (h, l, b, f) e discendenti (g, p, q).",
    ),
    (
        "Spaziatura grafica",
        "La spaziatura riguarda la distanza tra lettere, parole e righe. Spaziatura ampia tra le parole "
        "indica bisogno di spazio personale, pensiero indipendente e, talvolta, solitudine. "
        "Spaziatura ridotta (parole quasi attaccate) è correlata a socievolezza eccessiva, difficoltà "
        "nei confini relazionali e, in casi estremi, pensiero confusionario. La spaziatura irregolare — "
        "alternanza di parole distanti e ravvicinate — è un indicatore di disorganizzazione cognitiva "
        "o di scrittura non spontanea (es. copiatura o dettatura lenta). In perizie forensi, "
        "la spaziatura viene misurata in millimetri su campioni standardizzati.",
    ),
    (
        "Margini e layout",
        "I margini del foglio riflettono il rapporto dello scrittore con l'ambiente e il contesto "
        "sociale. Un margine sinistro ampio e costante indica rispetto delle regole e pianificazione. "
        "Un margine sinistro che si allarga progressivamente (testo che 'scivola' verso destra) "
        "suggerisce entusiasmo crescente o impulsività. Margine destro ampio è associato a prudenza, "
        "timore del futuro e riservatezza. L'assenza di margini (testo che occupa tutto il foglio) "
        "indica esuberanza comunicativa o senso di urgenza. In perizia, il margine aiuta a "
        "distinguere scritti autentici da trascrizioni o copie, poiché l'autore mantiene "
        "inconsciamente le proprie abitudini spaziali.",
    ),
    (
        "Firme autentiche",
        "Una firma autentica possiede caratteristiche di naturalezza e fluidità del movimento. "
        "I tratti sono continui, con accelerazione e decelerazione tipiche del gesto automatizzato. "
        "La pressione varia in modo coerente con il ritmo del tratto. I legamenti tra le lettere "
        "sono coerenti con il corpus grafico dello scrittore. La firma autentica presenta micro-tremori "
        "naturali (diversi dai tremori patologici) e piccole variazioni tra esecuzioni successive, "
        "mai perfettamente identiche. In perizia calligrafica, si confrontano almeno 10-15 firme "
        "autentiche per stabilire la 'gamma di variazione naturale' prima di esaminare la firma contestata.",
    ),
    (
        "Firme false",
        "Le firme contraffatte si distinguono per diversi indicatori: velocità di esecuzione "
        "innaturalmente lenta (visibile nei 'tocchi' del pennino e nelle esitazioni), tremori "
        "artificiali (regolari, non spontanei), ritocchi e correzioni del tratto, interruzioni "
        "anomale del gesto. La falsificazione per imitazione diretta (calco o copia visiva) produce "
        "una firma con aspetto simile all'originale ma con movimenti invertiti rispetto alla direzione "
        "naturale. Il falsario tende a concentrarsi sulla forma complessiva trascurando i dettagli "
        "minuti (proporzioni tra lettere, angolo di attacco del tratto, pressione). "
        "L'analisi forense utilizza ingrandimenti 10x-40x e, nei casi dubbi, grafometria digitale.",
    ),
    (
        "Velocità e ritmo",
        "La velocità di scrittura si manifesta nella forma delle lettere (semplificazione dei tratti "
        "in scrittura rapida), nell'inclinazione (più marcata ad alta velocità), nelle legature "
        "(frequenti in scrittura veloce, assenti in quella lenta). Il ritmo è la regolarità con cui "
        "si alternano tensione e distensione nel movimento grafico. Un ritmo regolare indica "
        "equilibrio psicofisico. Un ritmo aritmico (alternanza caotica di tratti tesi e distesi) "
        "può segnalare stati emotivi alterati, patologie neurologiche o scrittura non spontanea. "
        "In perizia forense la velocità è cruciale: una firma depositata 'lentamente' da una persona "
        "abitualmente veloce è un forte indicatore di contraffazione.",
    ),
    (
        "Datazione documenti",
        "La datazione grafica di un documento si basa su elementi intrinseci ed estrinseci. "
        "Elementi intrinseci: evoluzione dello stile grafico dell'autore nel tempo (campioni noti "
        "datati permettono di costruire una 'curva di evoluzione'), deterioramento della calligrafia "
        "legato all'età, variazioni nelle abitudini punteggiatura e abbreviazioni. "
        "Elementi estrinseci: tipo di inchiostro (analisi spettroscopica), supporto cartaceo "
        "(filigrana, composizione chimica), strumento di scrittura (biro, stilografica, matita). "
        "L'analisi dell'inchiostro mediante cromatografia liquida può stabilire se l'inchiostro "
        "è compatibile con la data dichiarata. In perizia, la datazione grafica va sempre "
        "abbinata ad analisi chimiche per raggiungere un grado di certezza forense.",
    ),
]

_RAG_KNOWLEDGE_DIR = ROOT / "data" / "knowledge"


def _chunk_text(text: str, source: str, size: int = 500, overlap: int = 50) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "source": source, "emb": None})
        start += size - overlap
    return chunks


def _ollama_embed(text: str):
    try:
        r = _requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": OLLAMA_MODEL, "prompt": text},
            timeout=30,
        )
        return np.array(r.json()["embedding"], dtype=np.float32)
    except Exception:
        return None


def _cosine_top_k(query_emb: np.ndarray, k: int = 3) -> list:
    if not _rag_chunks:
        return []
    embs = np.stack([c["emb"] for c in _rag_chunks if c["emb"] is not None])
    valid = [c for c in _rag_chunks if c["emb"] is not None]
    if len(valid) == 0:
        return []
    q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    scores = (embs / norms) @ q
    idxs = np.argsort(scores)[::-1][:k]
    return [(float(scores[i]), valid[i]) for i in idxs]


def _rag_cache_path(filename: str, file_bytes: bytes) -> Path:
    h = _hashlib.sha256(file_bytes).hexdigest()[:8]
    stem = Path(filename).stem[:40]
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return _RAG_CACHE_DIR / f"{safe}_{h}.npz"


def _rag_cache_save(cache_path: Path, chunks: list, filename: str) -> None:
    _RAG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    good = [c for c in chunks if c["emb"] is not None]
    if not good:
        return
    texts = np.array([c["text"] for c in good], dtype=object)
    sources = np.array([c["source"] for c in good], dtype=object)
    embs = np.stack([c["emb"] for c in good])
    np.savez_compressed(
        str(cache_path),
        texts=texts,
        sources=sources,
        embs=embs,
        filename=np.array(filename, dtype=object),
    )


def _rag_cache_load(cache_path: Path) -> tuple:
    """Returns (chunks, original_filename)."""
    data = np.load(str(cache_path), allow_pickle=True)
    filename = str(data["filename"])
    chunks = [
        {"text": str(t), "source": str(s), "emb": e}
        for t, s, e in zip(data["texts"], data["sources"], data["embs"])
    ]
    return chunks, filename


def _rag_doc_list() -> list:
    """Return rows [[filename, chunk_count]] for gr.Dataframe (user docs only)."""
    synthetic_sources = {s for s, _ in _RAG_SYNTHETIC_DOCS}
    counts: dict = {}
    for c in _rag_chunks:
        src = c["source"]
        if src not in synthetic_sources:
            counts[src] = counts.get(src, 0) + 1
    return [[name, cnt] for name, cnt in sorted(counts.items())]


def _rag_doc_choices() -> list:
    return [row[0] for row in _rag_doc_list()]


def _extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF, falling back to EasyOCR for scanned pages."""
    full_text = []
    try:
        import pypdf
    except ImportError:
        print(f"[RAG] pypdf not installed — skipping {path.name}")
        return ""
    try:
        reader = pypdf.PdfReader(str(path))
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if len(page_text.strip()) >= 50:
                full_text.append(page_text)
            else:
                # Scanned page — render to image and OCR
                try:
                    import fitz  # pymupdf
                    doc = fitz.open(str(path))
                    fitz_page = doc[page_num]
                    mat = fitz.Matrix(150 / 72, 150 / 72)  # 150 DPI
                    pix = fitz_page.get_pixmap(matrix=mat)
                    img_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.height, pix.width, pix.n
                    )
                    if pix.n == 4:
                        img_arr = img_arr[:, :, :3]
                    ocr_result = get_easyocr().readtext(img_arr, detail=0, paragraph=True)
                    full_text.append(" ".join(ocr_result))
                    doc.close()
                except ImportError:
                    print(f"[RAG] pymupdf not installed — cannot OCR scanned page {page_num+1} of {path.name}")
                except Exception as e:
                    print(f"[RAG] OCR error on page {page_num+1} of {path.name}: {e}")
    except Exception as e:
        print(f"[RAG] Error reading PDF {path.name}: {e}")
    return "\n".join(full_text)


def _rag_load_docs():
    global _rag_chunks, _rag_indexed_files, _rag_ready
    with _rag_lock:
        chunks: list = []

        # Synthetic built-in knowledge (always re-embedded at startup)
        for source, text in _RAG_SYNTHETIC_DOCS:
            chunks.extend(_chunk_text(text, source))

        # Load cached user documents (pre-embedded — no Ollama calls needed)
        _RAG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for cache_file in sorted(_RAG_CACHE_DIR.glob("*.npz")):
            try:
                cached_chunks, orig_filename = _rag_cache_load(cache_file)
                chunks.extend(cached_chunks)
                _rag_indexed_files.add(orig_filename)
                print(f"[RAG] Loaded from cache: {orig_filename} ({len(cached_chunks)} chunks)")
            except Exception as e:
                print(f"[RAG] Corrupt cache file {cache_file.name}: {e} — skipping")

        _rag_chunks = chunks
        _rag_ready = True
        print(f"[RAG] Chunks loaded: {len(chunks)} (synthetic + cached)")

    # Embed only synthetic chunks (emb is None); cached chunks already have embeddings
    embedded = 0
    for chunk in _rag_chunks:
        if chunk["emb"] is None:
            emb = _ollama_embed(chunk["text"])
            if emb is not None:
                chunk["emb"] = emb
                embedded += 1
    print(f"[RAG] Synthetic embedding done: {embedded} chunks")


def rag_add_docs(files) -> tuple:
    """Index uploaded PDF/DOCX files and add them to the live knowledge base."""
    global _rag_indexed_files
    if not files:
        return "Nessun file caricato.", _rag_doc_list()
    try:
        _requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
    except Exception:
        return (
            "❌ Ollama non raggiungibile — i documenti non possono essere indicizzati.\n"
            "Avvia `ollama serve` e ricarica.",
            _rag_doc_list(),
        )
    lines = []
    for f in files:
        path = Path(f.name)
        suffix = path.suffix.lower()
        if path.name in _rag_indexed_files:
            lines.append(f"ℹ️ `{path.name}` — già indicizzato, saltato.")
            continue

        file_bytes = path.read_bytes()
        cache_path = _rag_cache_path(path.name, file_bytes)

        # Load from cache if available (avoids re-embedding)
        if cache_path.exists():
            try:
                cached_chunks, _ = _rag_cache_load(cache_path)
                with _rag_lock:
                    _rag_chunks.extend(cached_chunks)
                    _rag_indexed_files.add(path.name)
                lines.append(f"✅ `{path.name}` — {len(cached_chunks)} chunk caricati dalla cache.")
                continue
            except Exception:
                pass  # fall through to re-embed

        try:
            if suffix == ".pdf":
                text = _extract_pdf_text(path)
            elif suffix in (".docx", ".doc"):
                import docx as _docx
                doc_obj = _docx.Document(str(path))
                text = "\n".join(p.text for p in doc_obj.paragraphs)
            else:
                lines.append(f"⚠️ `{path.name}` — formato non supportato (solo PDF/DOCX).")
                continue
        except Exception as e:
            lines.append(f"❌ `{path.name}` — errore: {e}")
            continue

        if not text.strip():
            lines.append(f"⚠️ `{path.name}` — nessun testo estratto.")
            continue

        chunks = _chunk_text(text, path.name)
        embedded = 0
        for chunk in chunks:
            emb = _ollama_embed(chunk["text"])
            if emb is not None:
                chunk["emb"] = emb
                embedded += 1

        try:
            _rag_cache_save(cache_path, chunks, path.name)
        except Exception as e:
            print(f"[RAG] Cache write failed for {path.name}: {e}")

        with _rag_lock:
            _rag_chunks.extend(chunks)
            _rag_indexed_files.add(path.name)
        lines.append(f"✅ `{path.name}` — {len(chunks)} chunk, {embedded} indicizzati.")

    return "\n".join(lines), _rag_doc_list()


def rag_remove_doc(filename: str) -> tuple:
    """Remove all chunks for a document from memory and delete its cache file."""
    global _rag_chunks, _rag_indexed_files
    if not filename or not filename.strip():
        return "Nessun documento selezionato.", _rag_doc_list()

    with _rag_lock:
        before = len(_rag_chunks)
        _rag_chunks = [c for c in _rag_chunks if c["source"] != filename]
        removed_chunks = before - len(_rag_chunks)
        _rag_indexed_files.discard(filename)

    deleted_files = 0
    if _RAG_CACHE_DIR.exists():
        for cache_file in _RAG_CACHE_DIR.glob("*.npz"):
            try:
                with np.load(str(cache_file), allow_pickle=True) as data:
                    match = str(data["filename"]) == filename
                if match:
                    cache_file.unlink()
                    deleted_files += 1
            except Exception:
                pass

    if removed_chunks == 0:
        return f"⚠️ `{filename}` non trovato nell'indice.", _rag_doc_list()

    msg = f"🗑️ `{filename}` rimosso ({removed_chunks} chunk eliminati"
    if deleted_files:
        msg += ", cache eliminata"
    msg += ")."
    return msg, _rag_doc_list()


def rag_query(question: str) -> str:
    if not question or not question.strip():
        return ""
    # Verifica Ollama raggiungibile
    try:
        _requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
    except Exception:
        return (
            "❌ **Ollama non raggiungibile.**\n\n"
            "Avvia il server con:\n```\nollama serve\n```\n"
            "e assicurati che il modello sia scaricato:\n"
            "```\nollama pull llama3.2\n```"
        )
    if not _rag_ready:
        return "⏳ Indice della knowledge base in costruzione, riprovare tra qualche secondo…"

    embedded_chunks = [c for c in _rag_chunks if c["emb"] is not None]
    if not embedded_chunks:
        total = len(_rag_chunks)
        return (
            f"⏳ Embedding in corso (0/{total} chunk pronti). "
            "Riprovare tra qualche secondo — l'indicizzazione procede in background."
        )

    q_emb = _ollama_embed(question)
    if q_emb is None:
        return "❌ Impossibile generare l'embedding della domanda. Ollama è in esecuzione?"

    synthetic_sources = {s for s, _ in _RAG_SYNTHETIC_DOCS}
    user_chunks = [c for c in _rag_chunks if c["emb"] is not None and c["source"] not in synthetic_sources]
    synth_chunks = [c for c in _rag_chunks if c["emb"] is not None and c["source"] in synthetic_sources]

    def _top_k_from(pool, q, k):
        if not pool:
            return []
        embs = np.stack([c["emb"] for c in pool])
        q_n = q / (np.linalg.norm(q) + 1e-9)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        scores = (embs / norms) @ q_n
        idxs = np.argsort(scores)[::-1][:k]
        return [(float(scores[i]), pool[i]) for i in idxs]

    # Always include up to 2 user-doc chunks + 2 synthetic chunks
    user_results = _top_k_from(user_chunks, q_emb, 2)
    synth_results = _top_k_from(synth_chunks, q_emb, 2)
    # If no user docs, fall back to more synthetic chunks
    if not user_results:
        synth_results = _top_k_from(synth_chunks, q_emb, 4)
    results = user_results + synth_results

    context = "\n\n".join(
        f"[{c['source']}]\n{c['text']}" for _, c in results
    )
    prompt = (
        "Sei un esperto di grafologia forense. Rispondi in italiano, in modo preciso e "
        "conciso, basandoti ESCLUSIVAMENTE sui seguenti estratti.\n\n"
        f"{context}\n\n"
        f"Domanda: {question}\n\nRisposta:"
    )
    try:
        r = _requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        answer = r.json().get("response", "").strip()
    except Exception as e:
        return f"❌ Errore nella generazione: {e}"

    sources = list(dict.fromkeys(c["source"] for _, c in results))
    return f"{answer}\n\n---\n*Fonti: {', '.join(sources)}*"


with gr.Blocks() as rag_tab:
    gr.Markdown(
        "## Consulente Forense IA\n"
        "Fai domande sulla grafologia forense. Il sistema recupera gli estratti più "
        "rilevanti dalla knowledge base e genera una risposta in italiano con "
        "**Llama 3.2 via Ollama** (locale, nessun dato inviato online)."
    )

    with gr.Accordion("📂 Gestione knowledge base", open=False):
        gr.Markdown(
            "Carica uno o più file PDF o DOCX per arricchire la knowledge base. "
            "Gli embedding vengono salvati su disco e ricaricati automaticamente al prossimo avvio.\n\n"
            "I PDF scansionati vengono trascritti automaticamente con OCR."
        )
        rag_upload = gr.File(
            label="Documenti (PDF o DOCX)",
            file_count="multiple",
            file_types=[".pdf", ".docx", ".doc"],
        )
        rag_upload_btn = gr.Button("Indicizza documenti", variant="secondary")
        rag_upload_status = gr.Markdown(label="Esito indicizzazione")

        gr.Markdown("### Documenti indicizzati")
        rag_doc_table = gr.Dataframe(
            headers=["Documento", "Chunk"],
            datatype=["str", "number"],
            interactive=False,
            label="Documenti nella knowledge base",
            value=_rag_doc_list,
        )
        with gr.Row():
            rag_remove_dd = gr.Dropdown(
                label="Seleziona documento da rimuovere",
                choices=_rag_doc_choices(),
                interactive=True,
            )
            rag_remove_btn = gr.Button("🗑️ Rimuovi", variant="secondary")
        rag_remove_status = gr.Markdown(label="Esito rimozione")

        rag_upload_btn.click(
            fn=rag_add_docs,
            inputs=rag_upload,
            outputs=[rag_upload_status, rag_doc_table],
        ).then(
            fn=lambda: gr.update(choices=_rag_doc_choices()),
            inputs=None,
            outputs=rag_remove_dd,
        )
        rag_remove_btn.click(
            fn=rag_remove_doc,
            inputs=rag_remove_dd,
            outputs=[rag_remove_status, rag_doc_table],
        ).then(
            fn=lambda: gr.update(choices=_rag_doc_choices(), value=None),
            inputs=None,
            outputs=rag_remove_dd,
        )

    rag_in = gr.Textbox(
        label="Domanda",
        placeholder="Es: Come si valuta l'inclinazione della scrittura?",
        lines=2,
    )
    rag_btn = gr.Button("Chiedi", variant="primary")
    rag_out = gr.Markdown(label="Risposta")
    gr.Examples(
        examples=[
            ["Cosa indica una forte pressione nella scrittura?"],
            ["Come si distingue una firma autentica da una contraffatta?"],
            ["Quali parametri grafologici rilevano stress o malattia?"],
            ["Come si data un documento manoscritto?"],
        ],
        inputs=rag_in,
    )
    rag_btn.click(rag_query, inputs=rag_in, outputs=rag_out)

    # Refresh table and dropdown when tab loads (background thread may have finished by then)
    rag_tab.load(
        fn=lambda: (gr.update(value=_rag_doc_list()), gr.update(choices=_rag_doc_choices())),
        outputs=[rag_doc_table, rag_remove_dd],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────────────────────────────────────

demo = gr.TabbedInterface(
    interface_list=[
        htr_tab, sig_verify_tab, sig_detect_tab,
        ner_tab, writer_tab, grapho_tab, pipeline_tab, dating_tab, rag_tab,
    ],
    tab_names=[
        "OCR Manoscritto",
        "Verifica Firma",
        "Rilevamento Firma",
        "Riconoscimento Entità",
        "Identificazione Scrittore",
        "Analisi Grafologica",
        "Pipeline Forense",
        "Datazione Documenti",
        "Consulente Forense IA",
    ],
    title=(
        "GraphoLab — Intelligenza Artificiale in Grafologia Forense"
        + ("\n⚠️ Demo su CPU: la prima inferenza per tab può richiedere 30–60 s."
           if os.environ.get("SPACE_ID") else "")
    ),
)

_threading.Thread(target=_rag_load_docs, daemon=True).start()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )
