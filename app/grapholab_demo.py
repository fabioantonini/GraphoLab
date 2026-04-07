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

import base64 as _base64
import io
import re as _re
import tempfile as _tempfile
import threading as _threading
from datetime import datetime as _datetime

import gradio as gr
import numpy as np
from PIL import Image

# ── core imports ──────────────────────────────────────────────────────────────
from core.ocr import htr_transcribe, get_easyocr
from core.ner import ner_extract
from core.graphology import grapho_analyse
from core.writer import writer_identify, ensure_writer_examples
from core.signature import sig_verify, sig_detect, detect_and_crop
from core.dating import dating_rank as _dating_rank_core, extract_dates
from core.pipeline import run_pipeline_steps, generate_forensic_pdf, PipelineResults
from core.rag import (
    check_ollama, ollama_list_models, set_rag_model,
    rag_load_docs, rag_add_docs, rag_remove_doc,
    rag_doc_list as _rag_doc_list, rag_doc_choices as _rag_doc_choices,
    rag_chat_stream, _rag_ready,
)
from core.agent import agent_stream, SUGGESTED_PROMPTS, AGENT_MODEL

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

SIGNET_WEIGHTS = ROOT / "models" / "signet.pth"
WRITER_SAMPLES_DIR = ROOT / "data" / "samples"
WRITER_EXAMPLES_DIR = WRITER_SAMPLES_DIR / "writer_examples"
RAG_CACHE_DIR = ROOT / "data" / "rag_cache"

from core.providers import openai_key_configured, invalidate_openai_client, is_openai_model
_OLLAMA_AVAILABLE = check_ollama()
_LLM_AVAILABLE = _OLLAMA_AVAILABLE or openai_key_configured()
_OPENAI_MODELS = ["gpt-5.4-mini", "gpt-5.4", "gpt-5.4-nano"]

# ──────────────────────────────────────────────────────────────────────────────
# Tab 1 — Handwritten OCR
# ──────────────────────────────────────────────────────────────────────────────

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


def _sig_verify_wrapper(ref_image, ref_image2, query_image):
    return sig_verify(ref_image, ref_image2, query_image, SIGNET_WEIGHTS)


_sig_examples = []
for _n in ["1", "2", "3"]:
    _r1 = _sig_ex(f"genuine_{_n}_1.png")
    _r2 = _sig_ex(f"genuine_{_n}_2.png")
    _forg = _sig_ex(f"forged_{_n}_1.png")
    if _r1 and _r2 and _forg:
        _sig_examples.append([_r1, _r2, _forg])
    if _n == "1" and _r1 and _r2:
        _sig_examples.append([_r1, None, _r2])


sig_verify_tab = gr.Interface(
    fn=_sig_verify_wrapper,
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

_writer_example_paths = ensure_writer_examples(WRITER_EXAMPLES_DIR)
_threading.Thread(
    target=lambda: writer_identify(None, WRITER_SAMPLES_DIR),  # pre-warm (returns early on None)
    daemon=True,
).start()


def _writer_identify_wrapper(image):
    return writer_identify(image, WRITER_SAMPLES_DIR)


writer_tab = gr.Interface(
    fn=_writer_identify_wrapper,
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

def run_pipeline(
    doc_image: np.ndarray,
    ref_sig: np.ndarray | None,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    """Gradio generator: yields partial UI outputs after each pipeline step."""
    _ = gr.update()

    if doc_image is None:
        msg = "Carica il documento da analizzare."
        yield (doc_image, msg, msg, [], msg, msg, None, msg, doc_image, msg, None, msg,
               gr.update(visible=False), _)
        return

    def _on_progress(step, total, desc):
        progress(step / total, desc=f"Step {step}/{total} — {desc}")

    results = PipelineResults()

    for results in run_pipeline_steps(
        doc_image, ref_sig, SIGNET_WEIGHTS, WRITER_SAMPLES_DIR, _on_progress
    ):
        yield (
            results.sig_detect_image, results.sig_detect_summary,
            results.htr_text,
            results.ner_highlighted, results.ner_summary,
            results.writer_report, results.writer_chart,
            results.grapho_report, results.grapho_image,
            results.sig_verify_report, results.sig_verify_chart,
            results.final_report,
            gr.update(visible=bool(results.sig_detect_summary)),
            results.llm_report,
        )


def _generate_pipeline_pdf_wrapper(
    s1_img, s1_txt, s2_txt, s3_md,
    s4_md, s4_img, s5_md, s5_img,
    s6_txt, s6_img, llm_text,
) -> str:
    results = PipelineResults(
        sig_detect_image=s1_img, sig_detect_summary=s1_txt or "",
        htr_text=s2_txt or "",
        ner_summary=s3_md or "",
        writer_report=s4_md or "", writer_chart=s4_img,
        grapho_report=s5_md or "", grapho_image=s5_img,
        sig_verify_report=s6_txt or "", sig_verify_chart=s6_img,
        llm_report=llm_text or "",
    )
    return generate_forensic_pdf(results)


with gr.Blocks() as pipeline_tab:
    gr.Markdown(
        "## Perizia Forense Automatica\n\n"
        "Carica il documento da esaminare (es. testamento olografo, lettera anonima, contratto) "
        "e, opzionalmente, una firma di riferimento autentica. "
        "Il sistema eseguirà in sequenza tutti e sei gli strumenti AI e produrrà un **referto forense integrato**.\n\n"
        "| Step | Strumento | Input |\n"
        "|------|-----------|-------|\n"
        "| 1 | Rilevamento Firma (Conditional DETR) | Documento |\n"
        "| 2 | Trascrizione HTR (EasyOCR) | Documento |\n"
        "| 3 | Riconoscimento Entità — NER | Testo da Step 2 |\n"
        "| 4 | Identificazione Scrittore | Documento |\n"
        "| 5 | Analisi Grafologica | Documento |\n"
        "| 6 | Verifica Firma (SigNet) | Firma rif. + crop da Step 1 |\n"
        "| 7 | Sintesi LLM (Ollama) | Output Step 1–6 |\n"
    )

    with gr.Row():
        pipe_doc = gr.Image(label="Documento da analizzare (testamento, lettera, atto)", type="numpy")
        pipe_ref = gr.Image(label="Firma di riferimento nota — opzionale (per Step 6)", type="numpy")

    pipe_btn = gr.Button("▶  Avvia Analisi Forense", variant="primary", size="lg")

    with gr.Column(visible=False) as pipe_results:
        gr.Markdown("### Step 1 — Rilevamento Firma (Conditional DETR)")
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

        with gr.Accordion("Step 7 — Valutazione LLM (Ollama)", open=True):
            out_llm = gr.Markdown(label="Referto sintetico LLM")

    with gr.Row(visible=False) as pdf_row:
        pdf_btn = gr.Button("📄 Scarica Report PDF", variant="secondary")
        pdf_out = gr.File(label="Report PDF", file_types=[".pdf"])

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
            out_llm,
        ],
    ).then(
        fn=lambda: gr.update(visible=True),
        inputs=None,
        outputs=pdf_row,
    )

    pdf_btn.click(
        fn=_generate_pipeline_pdf_wrapper,
        inputs=[
            out_s1_img, out_s1_txt,
            out_s2_txt,
            out_s3_md,
            out_s4_md, out_s4_img,
            out_s5_md, out_s5_img,
            out_s6_txt, out_s6_img,
            out_llm,
        ],
        outputs=pdf_out,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Tab 8 — Document Dating
# ──────────────────────────────────────────────────────────────────────────────

def _dating_rank_gradio(files: list) -> str:
    """Gradio wrapper: converts gr.File objects to file paths for core function."""
    if not files:
        return "Carica almeno un'immagine di documento."
    paths = [f.name if hasattr(f, "name") else str(f) for f in files]
    return _dating_rank_core(paths)


dating_tab = gr.Interface(
    fn=_dating_rank_gradio,
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

def _content_str(content) -> str:
    """Normalize Gradio 6.x content field (str or list of parts) to plain str."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", "") or part.get("content", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content) if content is not None else ""


def rag_chat(message: str, history: list):
    """Gradio streaming wrapper for rag_chat_stream."""
    if not message or not message.strip():
        yield history
        return

    # Normalise history content to plain strings for core function
    normalised_history = [
        {"role": msg["role"], "content": _content_str(msg["content"])}
        for msg in history
    ]

    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]

    try:
        for partial, sources_footer in rag_chat_stream(message, normalised_history):
            content = partial + (sources_footer or "")
            new_history[-1]["content"] = content
            yield new_history
    except Exception as e:
        new_history[-1]["content"] = f"❌ Errore: {e}"
        yield new_history


def _apply_openai_key(key: str) -> str:
    """Set OpenAI API key in the process environment and switch RAG/embed models to OpenAI.

    Returns a status message string.
    """
    key = key.strip()
    if not key:
        return "⚠️ Chiave vuota — inserisci una chiave valida (es. sk-...)."
    os.environ["OPENAI_API_KEY"] = key
    invalidate_openai_client()
    from core.rag import set_rag_model, _embed_model, set_embed_model
    set_rag_model("gpt-5.4-mini")
    if not is_openai_model(_embed_model):
        set_embed_model("text-embedding-3-small")
    return "✅ Chiave salvata. Modello: **gpt-5.4-mini** · Embedding: **text-embedding-3-small**"


def _rag_add_docs_wrapper(files):
    return rag_add_docs(files, RAG_CACHE_DIR)


def _rag_remove_doc_wrapper(filename):
    return rag_remove_doc(filename, RAG_CACHE_DIR)


def save_conversation_md(history: list):
    """Save current chat history as a Markdown file and return path for download."""
    if not history:
        return gr.update(visible=False)
    now = _datetime.now()
    lines = [f"# Conversazione Forense — {now.strftime('%Y-%m-%d %H:%M')}\n"]
    for msg in history:
        role = msg.get("role", "")
        content = _content_str(msg.get("content", ""))
        if role == "user":
            lines.append(f"**Utente:** {content}\n")
        elif role == "assistant":
            lines.append(f"**Consulente:** {content}\n")
            lines.append("---\n")
    filename = f"conversazione_{now.strftime('%Y%m%d_%H%M%S')}.md"
    filepath = Path(_tempfile.gettempdir()) / filename
    filepath.write_text("\n".join(lines), encoding="utf-8")
    return gr.update(value=str(filepath), visible=True)


with gr.Blocks() as rag_tab:
    gr.Markdown(
        "## Consulente Forense IA\n"
        "Fai domande sulla grafologia forense. Il sistema recupera gli estratti più "
        "rilevanti dalla knowledge base e genera una risposta in italiano.\n\n"
        "Funziona con **Ollama locale** (nessun dato inviato online) oppure con "
        "**OpenAI** inserendo la tua chiave nell'accordion qui sotto."
    )

    with gr.Accordion("⚙️ Configurazione LLM", open=not _LLM_AVAILABLE):
        if _OLLAMA_AVAILABLE:
            gr.Markdown("✅ **Ollama disponibile** — i modelli locali sono in uso.")
        else:
            gr.Markdown(
                "Ollama non rilevato. Inserisci la tua chiave OpenAI per abilitare "
                "il Consulente Forense IA su questa demo.\n\n"
                "*La chiave viene usata solo per questa sessione e non viene salvata su disco.*"
            )
        rag_openai_key = gr.Textbox(
            label="Chiave OpenAI (sk-...)",
            type="password",
            placeholder="sk-...",
            visible=not _OLLAMA_AVAILABLE,
        )
        rag_openai_save_btn = gr.Button(
            "Salva chiave e abilita",
            variant="secondary",
            visible=not _OLLAMA_AVAILABLE,
        )
        rag_openai_status = gr.Markdown(visible=not _OLLAMA_AVAILABLE)

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
        rag_upload_btn = gr.Button("Indicizza documenti", variant="secondary", interactive=_LLM_AVAILABLE)
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
            fn=_rag_add_docs_wrapper,
            inputs=rag_upload,
            outputs=[rag_upload_status, rag_doc_table],
        ).then(
            fn=lambda: gr.update(choices=_rag_doc_choices()),
            inputs=None,
            outputs=rag_remove_dd,
        )
        rag_remove_btn.click(
            fn=_rag_remove_doc_wrapper,
            inputs=rag_remove_dd,
            outputs=[rag_remove_status, rag_doc_table],
        ).then(
            fn=lambda: gr.update(choices=_rag_doc_choices(), value=None),
            inputs=None,
            outputs=rag_remove_dd,
        )

    _rag_initial_choices = (
        (_OPENAI_MODELS if openai_key_configured() else []) +
        (ollama_list_models() if _OLLAMA_AVAILABLE else [])
    ) or _OPENAI_MODELS
    _rag_initial_value = (
        "gpt-5.4-mini" if openai_key_configured()
        else (ollama_list_models()[0] if _OLLAMA_AVAILABLE else "gpt-5.4-mini")
    )
    with gr.Row():
        rag_model_dd = gr.Dropdown(
            label="Modello di generazione",
            choices=_rag_initial_choices,
            value=_rag_initial_value,
            interactive=True,
            scale=3,
        )
        rag_model_refresh = gr.Button("🔄", variant="secondary", scale=1)
    rag_model_status = gr.Markdown(visible=False)
    gr.Markdown(
        "*⚠️ Cambiare modello dopo aver indicizzato documenti può degradare il retrieval "
        "(gli embedding in cache usano il modello precedente). "
        "Re-indicizzare i documenti per risultati ottimali.*"
    )

    rag_chatbot = gr.Chatbot(label="Consulente Forense IA", height=500)
    rag_in = gr.Textbox(
        placeholder=(
            "Es: Come si valuta l'inclinazione della scrittura? (Invio per inviare)"
            if _LLM_AVAILABLE
            else "⚠️ Inserisci la chiave OpenAI nell'accordion ⚙️ qui sopra per abilitare"
        ),
        lines=1,
        show_label=False,
        interactive=_LLM_AVAILABLE,
    )
    with gr.Row():
        rag_btn = gr.Button("Invia", variant="primary", interactive=_LLM_AVAILABLE)
        rag_clear_btn = gr.Button("🗑️ Cancella", variant="secondary")
        rag_save_btn = gr.Button("💾 Salva conversazione", variant="secondary")
    rag_download = gr.File(label="Download conversazione", visible=False)

    gr.Examples(
        examples=[
            ["Cosa indica una forte pressione nella scrittura?"],
            ["Come si distingue una firma autentica da una contraffatta?"],
            ["Quali parametri grafologici rilevano stress o malattia?"],
            ["Come si data un documento manoscritto?"],
        ],
        inputs=rag_in,
    )

    def _respond(message, history):
        for updated_history in rag_chat(message, history):
            yield "", updated_history

    def _rag_save_key(key):
        msg = _apply_openai_key(key)
        ok = "✅" in msg
        new_choices = _OPENAI_MODELS + (ollama_list_models() if _OLLAMA_AVAILABLE else [])
        return (
            msg,
            gr.update(interactive=ok, placeholder="Es: Come si valuta l'inclinazione della scrittura?"),
            gr.update(interactive=ok),
            gr.update(interactive=ok),
            gr.update(choices=new_choices, value="gpt-5.4-mini") if ok else gr.update(),
        )

    rag_openai_save_btn.click(
        fn=_rag_save_key,
        inputs=rag_openai_key,
        outputs=[rag_openai_status, rag_in, rag_btn, rag_upload_btn, rag_model_dd],
    )

    rag_model_dd.change(
        fn=lambda m: gr.update(value=set_rag_model(m), visible=True),
        inputs=rag_model_dd,
        outputs=rag_model_status,
    )

    def _rag_refresh_models():
        choices = (
            (_OPENAI_MODELS if openai_key_configured() else []) +
            (ollama_list_models() if _OLLAMA_AVAILABLE else [])
        ) or _OPENAI_MODELS
        value = choices[0] if choices else "gpt-5.4-mini"
        return gr.update(choices=choices, value=value)

    rag_model_refresh.click(fn=_rag_refresh_models, outputs=rag_model_dd)

    rag_btn.click(_respond, inputs=[rag_in, rag_chatbot], outputs=[rag_in, rag_chatbot])
    rag_in.submit(_respond, inputs=[rag_in, rag_chatbot], outputs=[rag_in, rag_chatbot])
    rag_clear_btn.click(
        fn=lambda: ([], "", gr.update(visible=False)),
        outputs=[rag_chatbot, rag_in, rag_download],
    )
    rag_save_btn.click(fn=save_conversation_md, inputs=rag_chatbot, outputs=rag_download)

    def _rag_tab_load():
        choices = (
            (_OPENAI_MODELS if openai_key_configured() else []) +
            (ollama_list_models() if _OLLAMA_AVAILABLE else [])
        ) or _OPENAI_MODELS
        value = choices[0] if choices else "gpt-5.4-mini"
        return (
            gr.update(value=_rag_doc_list()),
            gr.update(choices=_rag_doc_choices()),
            gr.update(choices=choices, value=value),
        )

    rag_tab.load(fn=_rag_tab_load, outputs=[rag_doc_table, rag_remove_dd, rag_model_dd])

# ──────────────────────────────────────────────────────────────────────────────
# Tab 11 — Agente Documentale
# ──────────────────────────────────────────────────────────────────────────────

with gr.Blocks() as agent_tab:
    gr.Markdown(
        "## Agente Documentale\n"
        "Descrivi in linguaggio naturale cosa vuoi fare e l'agente sceglierà automaticamente "
        "quali strumenti GraphoLab usare: trascrizione, NER, rilevamento firma, verifica firma, "
        "grafologia, identificazione scrittore, layout, datazione e molto altro.\n\n"
        "Allega uno o più file (immagini o PDF) e scrivi la tua richiesta.\n\n"
        "*Motore: LangGraph + Ollama locale oppure OpenAI (configura la chiave qui sotto)*"
    )

    with gr.Accordion("⚙️ Configurazione LLM", open=not _LLM_AVAILABLE):
        if _OLLAMA_AVAILABLE:
            gr.Markdown("✅ **Ollama disponibile** — i modelli locali sono in uso.")
        else:
            gr.Markdown(
                "Ollama non rilevato. Inserisci la tua chiave OpenAI per abilitare "
                "l'Agente Documentale su questa demo.\n\n"
                "*La chiave viene usata solo per questa sessione e non viene salvata su disco.*"
            )
        agent_openai_key = gr.Textbox(
            label="Chiave OpenAI (sk-...)",
            type="password",
            placeholder="sk-...",
            visible=not _OLLAMA_AVAILABLE,
        )
        agent_openai_save_btn = gr.Button(
            "Salva chiave e abilita",
            variant="secondary",
            visible=not _OLLAMA_AVAILABLE,
        )
        agent_openai_status = gr.Markdown(visible=not _OLLAMA_AVAILABLE)

    # Suggested prompt buttons
    gr.Markdown("### Prompt suggeriti")
    with gr.Row(elem_classes=["agent-prompts"]):
        _prompt_btns = []
        for _p in SUGGESTED_PROMPTS:
            _btn = gr.Button(
                _p["label"],
                variant="secondary",
                size="sm",
                interactive=_LLM_AVAILABLE,
            )
            _prompt_btns.append((_btn, _p["text"]))

    # File upload + chat
    with gr.Row():
        with gr.Column(scale=3):
            agent_chatbot = gr.Chatbot(
                label="Agente Documentale",
                height=480,
            )
            agent_input = gr.Textbox(
                placeholder=(
                    "Es: Trascrivi il testo e cerca le persone nominate (Invio per inviare)"
                    if _LLM_AVAILABLE
                    else "⚠️ Inserisci la chiave OpenAI nell'accordion ⚙️ qui sopra per abilitare"
                ),
                lines=2,
                show_label=False,
                interactive=_LLM_AVAILABLE,
            )
            with gr.Row():
                agent_send_btn = gr.Button(
                    "Invia", variant="primary", interactive=_LLM_AVAILABLE
                )
                agent_stop_btn = gr.Button("⏹ Stop", variant="stop")
                agent_clear_btn = gr.Button("🗑️ Cancella", variant="secondary")

        with gr.Column(scale=1):
            agent_files = gr.File(
                label="File allegati (immagini, PDF)",
                file_count="multiple",
                file_types=["image", ".pdf"],
                interactive=_LLM_AVAILABLE,
            )

    def _agent_save_key(key):
        msg = _apply_openai_key(key)
        ok = "✅" in msg
        _prompt_updates = [gr.update(interactive=ok)] * len(_prompt_btns)
        return (
            [msg,
             gr.update(interactive=ok, placeholder="Es: Trascrivi il testo e cerca le persone nominate"),
             gr.update(interactive=ok),
             gr.update(interactive=ok)]
            + _prompt_updates
        )

    agent_openai_save_btn.click(
        fn=_agent_save_key,
        inputs=agent_openai_key,
        outputs=(
            [agent_openai_status, agent_input, agent_send_btn, agent_files]
            + [btn for btn, _ in _prompt_btns]
        ),
    )

    # Wire prompt buttons: each button fills the input text
    for _btn, _text in _prompt_btns:
        _btn.click(fn=lambda t=_text: t, outputs=agent_input)

    _api_img_re = _re.compile(r'!\[(.*?)\]\(/api/agent/images/([^\)]+)\)')

    def _fix_agent_image_urls(text: str) -> str:
        """Replace /api/agent/images/<name> URLs with base64 data URIs for Gradio rendering."""
        def _to_b64(m):
            alt, fname = m.group(1), m.group(2)
            fpath = ROOT / "data" / "uploads" / "agent" / "images" / fname
            if fpath.exists():
                b64 = _base64.b64encode(fpath.read_bytes()).decode()
                return f"![{alt}](data:image/png;base64,{b64})"
            return m.group(0)
        return _api_img_re.sub(_to_b64, text)

    def _agent_respond(message, history, files):
        """Gradio generator: yield (input_clear, updated_history) on each stream event.

        History format: list of {role, content} dicts (Gradio 5 messages format).
        """
        if not message or not message.strip():
            yield "", history
            return

        # Collect file paths from Gradio File component (list of dicts or paths)
        file_paths = []
        if files:
            for f in files:
                path = f if isinstance(f, str) else (f.get("name") if isinstance(f, dict) else str(f))
                if path:
                    file_paths.append(path)

        # history is already in messages format — pass as-is to agent
        agent_history = [
            {"role": msg["role"], "content": _content_str(msg["content"])}
            for msg in (history or [])
        ]

        # Add user turn and placeholder assistant turn
        updated_history = list(history or []) + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": ""},
        ]

        for text in agent_stream(message, file_paths, agent_history):
            updated_history[-1] = {"role": "assistant", "content": _fix_agent_image_urls(text)}
            yield "", updated_history

    _send_event = agent_send_btn.click(
        fn=_agent_respond,
        inputs=[agent_input, agent_chatbot, agent_files],
        outputs=[agent_input, agent_chatbot],
    )
    _submit_event = agent_input.submit(
        fn=_agent_respond,
        inputs=[agent_input, agent_chatbot, agent_files],
        outputs=[agent_input, agent_chatbot],
    )
    agent_stop_btn.click(fn=None, cancels=[_send_event, _submit_event])
    agent_clear_btn.click(
        fn=lambda: ([], "", None),
        outputs=[agent_chatbot, agent_input, agent_files],
    )

# ──────────────────────────────────────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────────────────────────────────────

demo = gr.TabbedInterface(
    interface_list=[
        htr_tab, sig_verify_tab, sig_detect_tab,
        ner_tab, writer_tab, grapho_tab, pipeline_tab, dating_tab, rag_tab,
        agent_tab,
    ],
    tab_names=[
        "OCR Manoscritto",
        "Verifica Firma",
        "Rilevamento Firma",
        "Riconoscimento Entità",
        "Identificazione Scrittore",
        "Analisi Grafologica",
        "Perizia Forense Automatica",
        "Datazione Documenti",
        "Consulente Forense IA",
        "🤖 Agente Documentale",
    ],
    title=(
        "GraphoLab — Intelligenza Artificiale in Grafologia Forense"
        + ("\n⚠️ Demo su CPU: la prima inferenza per tab può richiedere 30–60 s."
           if os.environ.get("SPACE_ID") else "")
    ),
)

_threading.Thread(target=lambda: rag_load_docs(RAG_CACHE_DIR), daemon=True).start()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )
