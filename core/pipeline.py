"""
GraphoLab core — Forensic Pipeline and PDF Report Generation.

Provides:
  - run_pipeline_steps()    run all 6 AI steps and return structured results
  - generate_forensic_pdf() generate a PDF forensic report from pipeline results
"""

from __future__ import annotations

import io
import re
import tempfile
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generator

import numpy as np


def _llm_provider_label() -> str:
    """Return 'OpenAI (gpt-5.4)' or 'Ollama (qwen3:4b)' based on the active model."""
    try:
        from core.rag import _rag_model
        from core.providers import is_openai_model
        provider = "OpenAI" if is_openai_model(_rag_model) else "Ollama"
        return f"{provider} · {_rag_model}"
    except Exception:
        return "LLM"


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResults:
    """Structured results from the forensic pipeline."""
    # Step 1 — Signature Detection
    sig_detect_image: np.ndarray | None = None
    sig_detect_summary: str = ""
    sig_crop: np.ndarray | None = None

    # Step 2 — HTR
    htr_text: str = ""

    # Step 3 — NER
    ner_highlighted: list = field(default_factory=list)
    ner_summary: str = ""

    # Step 4 — Writer Identification
    writer_report: str = ""
    writer_chart: np.ndarray | None = None

    # Step 5 — Graphological Analysis
    grapho_report: str = ""
    grapho_image: np.ndarray | None = None

    # Step 6 — Signature Verification
    sig_verify_report: str = ""
    sig_verify_chart: np.ndarray | None = None

    # Final integrated report
    final_report: str = ""

    # Step 7 — LLM Synthesis
    llm_report: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline_steps(
    doc_image: np.ndarray,
    ref_sig: np.ndarray | None,
    signet_weights: Path,
    writer_samples_dir: Path,
    on_progress: callable | None = None,
) -> Generator[PipelineResults, None, None]:
    """Run all forensic pipeline steps, yielding partial results after each step.

    Args:
        doc_image:          Document image (RGB numpy array).
        ref_sig:            Optional reference signature for step 6.
        signet_weights:     Path to signet.pth weights.
        writer_samples_dir: Path to data/samples/ for writer model.
        on_progress:        Optional callback(step: int, total: int, desc: str).

    Yields:
        PipelineResults after each step (progressively filled).
    """
    from core.signature import detect_and_crop, sig_verify
    from core.ocr import htr_transcribe
    from core.ner import ner_extract
    from core.writer import writer_identify
    from core.graphology import grapho_analyse
    from core.rag import pipeline_llm_synthesis

    results = PipelineResults()
    total = 7

    def _progress(step: int, desc: str):
        if on_progress:
            on_progress(step, total, desc)

    # Step 1 — Signature Detection
    _progress(1, "Rilevamento firma…")
    results.sig_detect_image, results.sig_crop, results.sig_detect_summary = detect_and_crop(doc_image)
    yield results

    # Step 2 — HTR
    _progress(2, "Trascrizione HTR…")
    results.htr_text = htr_transcribe(doc_image)
    yield results

    # Step 3 — NER
    _progress(3, "Riconoscimento entità…")
    text_for_ner = results.htr_text if results.htr_text and results.htr_text.strip() else ""
    if text_for_ner:
        results.ner_highlighted, results.ner_summary = ner_extract(text_for_ner)
    else:
        results.ner_highlighted = []
        results.ner_summary = "Nessun testo trascritto disponibile per il NER."
    yield results

    # Step 4 — Writer Identification
    _progress(4, "Identificazione scrittore…")
    results.writer_report, results.writer_chart = writer_identify(doc_image, writer_samples_dir)
    yield results

    # Step 5 — Graphological Analysis
    _progress(5, "Analisi grafologica…")
    results.grapho_report, results.grapho_image = grapho_analyse(doc_image)
    yield results

    # Step 6 — Signature Verification
    _progress(6, "Verifica firma…")
    if ref_sig is not None:
        query_for_verify = results.sig_crop if results.sig_crop is not None else doc_image
        results.sig_verify_report, results.sig_verify_chart = sig_verify(
            ref_sig, None, query_for_verify, signet_weights
        )
        if results.sig_crop is None:
            results.sig_verify_report += "\n\n⚠️ Nessuna firma estratta — confronto eseguito sull'immagine intera."
    else:
        results.sig_verify_report = (
            "Firma di riferimento non fornita.\n\n"
            "Per abilitare questo step carica una firma autentica nota "
            "nel campo 'Firma di riferimento' sopra."
        )
        results.sig_verify_chart = None
    yield results

    # Integrated report
    results.final_report = (
        "## Referto Forense Integrato\n\n"
        "---\n\n"
        f"### Step 1 — Rilevamento Firma\n{results.sig_detect_summary}\n\n"
        f"### Step 2 — Trascrizione HTR\n```\n{results.htr_text}\n```\n\n"
        f"### Step 3 — Entità Nominate\n{results.ner_summary}\n\n"
        f"### Step 4 — Identificazione Scrittore\n{results.writer_report}\n\n"
        f"### Step 5 — Caratteristiche Grafologiche\n{results.grapho_report}\n\n"
        f"### Step 6 — Verifica Firma\n{results.sig_verify_report}\n\n"
        "---\n\n"
        "*Referto generato automaticamente da GraphoLab. "
        "Tutti i risultati hanno carattere indicativo e devono essere valutati "
        "da un perito calligrafo qualificato.*"
    )
    yield results

    # Step 7 — LLM Synthesis
    _progress(7, "Sintesi LLM…")
    results.llm_report = pipeline_llm_synthesis(
        results.sig_detect_summary,
        results.htr_text,
        results.ner_summary,
        results.writer_report,
        results.grapho_report,
        results.sig_verify_report,
    )

    # Update final_report to include LLM synthesis
    results.final_report = (
        "## Referto Forense Integrato\n\n"
        "---\n\n"
        f"### Step 1 — Rilevamento Firma\n{results.sig_detect_summary}\n\n"
        f"### Step 2 — Trascrizione HTR\n```\n{results.htr_text}\n```\n\n"
        f"### Step 3 — Entità Nominate\n{results.ner_summary}\n\n"
        f"### Step 4 — Identificazione Scrittore\n{results.writer_report}\n\n"
        f"### Step 5 — Caratteristiche Grafologiche\n{results.grapho_report}\n\n"
        f"### Step 6 — Verifica Firma\n{results.sig_verify_report}\n\n"
        f"### Step 7 — Valutazione LLM ({_llm_provider_label()})\n{results.llm_report}\n\n"
        "---\n\n"
        "*Referto generato automaticamente da GraphoLab. "
        "Tutti i risultati hanno carattere indicativo e devono essere valutati "
        "da un perito calligrafo qualificato.*"
    )
    yield results


# ──────────────────────────────────────────────────────────────────────────────
# PDF report generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_forensic_pdf(results: PipelineResults) -> str:
    """Generate a PDF forensic report from pipeline results. Returns the file path."""
    from fpdf import FPDF
    from PIL import Image as _PILImage

    def _to_latin1(text: str) -> str:
        if not text:
            return ""
        replacements = {
            "\u2014": "-", "\u2013": "-",
            "\u2018": "'", "\u2019": "'",
            "\u201c": '"', "\u201d": '"',
            "\u2026": "...",
            "\u2022": "*",
            "\u2713": "v", "\u2714": "v",
            "\u2718": "x", "\u2716": "x",
            "\U0001f947": "1.", "\U0001f948": "2.", "\U0001f949": "3.",
            "\u26a0\ufe0f": "(!)", "\u26a0": "(!)",
            "\U0001f50d": "",
            "\U0001f5d1": "",
        }
        for src, dst in replacements.items():
            text = text.replace(src, dst)
        return text.encode("latin-1", errors="replace").decode("latin-1")

    def _md_to_plain(text: str) -> str:
        if not text:
            return ""
        def _table_row_to_plain(m):
            cells = [c.strip() for c in m.group(0).strip("|").split("|")]
            return "  |  ".join(c for c in cells if c)
        text = re.sub(r"^[-| ]+$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\|.*\|$", _table_row_to_plain, text, flags=re.MULTILINE)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)
        text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return _to_latin1(text.strip())

    def _numpy_to_jpeg_bytes(arr) -> bytes | None:
        if arr is None:
            return None
        try:
            img = _PILImage.fromarray(arr.astype("uint8"))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return buf.getvalue()
        except Exception:
            return None

    class ForensicPDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 10)
            self.set_text_color(80, 80, 80)
            self.cell(0, 8, "GraphoLab - Referto Forense Integrato", align="C")
            self.ln(2)
            self.set_draw_color(180, 180, 180)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(130, 130, 130)
            self.cell(0, 10, f"Pagina {self.page_no()} - Generato da GraphoLab", align="C")

    pdf = ForensicPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 12, "Referto Forense Integrato", align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    pdf.cell(0, 8, f"Data generazione: {now}", align="C")
    pdf.ln(10)

    def _section_title(title: str):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(255, 255, 255)
        pdf.set_fill_color(50, 80, 120)
        pdf.cell(0, 8, _to_latin1(f"  {title}"), fill=True)
        pdf.ln(12)
        pdf.set_text_color(30, 30, 30)

    def _body_text(text: str):
        if not text:
            return
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 5, _md_to_plain(text))
        pdf.ln(3)

    def _llm_text(text: str):
        """Render LLM Markdown report with styled headings and bullet points."""
        if not text:
            return
        _heading_re = re.compile(r"^(#{1,6})\s+(.*)")
        _bullet_re  = re.compile(r"^[-*]\s+(.*)")
        _bold_re    = re.compile(r"\*{1,2}(.+?)\*{1,2}")

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                pdf.ln(2)
                continue

            m_heading = _heading_re.match(stripped)
            m_bullet  = _bullet_re.match(stripped)

            if m_heading:
                level   = len(m_heading.group(1))
                heading = _bold_re.sub(r"\1", m_heading.group(2))
                if level >= 3:          # H3/H4/H5/H6 → teal
                    pdf.set_font("Helvetica", "B", 10)
                    pdf.set_text_color(30, 100, 120)
                    pdf.multi_cell(0, 6, _to_latin1(heading))
                else:                   # H1/H2 → dark blue
                    pdf.set_font("Helvetica", "B", 11)
                    pdf.set_text_color(50, 80, 120)
                    pdf.multi_cell(0, 6, _to_latin1(heading))
                pdf.set_text_color(40, 40, 40)
            elif m_bullet:
                content = _bold_re.sub(r"\1", m_bullet.group(1))
                pdf.set_font("Helvetica", "", 10)
                pdf.set_x(pdf.get_x() + 4)
                pdf.multi_cell(0, 5, _to_latin1(f"\u2022  {content}"))
            elif stripped.startswith("**") and stripped.endswith("**"):
                content = _bold_re.sub(r"\1", stripped)
                pdf.set_font("Helvetica", "B", 10)
                pdf.multi_cell(0, 5, _to_latin1(content))
            else:
                content = _bold_re.sub(r"\1", stripped)
                pdf.set_font("Helvetica", "", 10)
                pdf.multi_cell(0, 5, _to_latin1(content))
        pdf.ln(3)

    def _embed_image(arr, max_w: int = 170):
        data = _numpy_to_jpeg_bytes(arr)
        if data is None:
            return
        buf = io.BytesIO(data)
        img = _PILImage.open(buf)
        w, h = img.size
        ratio = min(max_w / w, 100 / h)
        disp_w, disp_h = w * ratio, h * ratio
        buf.seek(0)
        x = (210 - disp_w) / 2
        pdf.image(buf, x=x, w=disp_w, h=disp_h)
        pdf.ln(4)

    _section_title("Step 1 — Rilevamento Firma (Conditional DETR)")
    _body_text(results.sig_detect_summary)
    _embed_image(results.sig_detect_image)

    _section_title("Step 2 — Trascrizione HTR (EasyOCR)")
    _body_text(results.htr_text)

    _section_title("Step 3 — Riconoscimento Entita' (NER)")
    _body_text(results.ner_summary or "Nessuna entita' rilevata nel testo trascritto.")

    _section_title("Step 4 — Identificazione Scrittore")
    _body_text(results.writer_report)
    _embed_image(results.writer_chart)

    _section_title("Step 5 — Analisi Grafologica")
    _body_text(results.grapho_report)
    _embed_image(results.grapho_image)

    _section_title("Step 6 — Verifica Firma (SigNet)")
    _body_text(results.sig_verify_report)
    _embed_image(results.sig_verify_chart)

    _section_title(f"Step 7 — Valutazione LLM ({_llm_provider_label()})")
    _llm_text(results.llm_report)

    pdf.ln(6)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(
        0, 4,
        "Referto generato automaticamente da GraphoLab. "
        "Tutti i risultati hanno carattere indicativo e devono essere valutati "
        "da un perito calligrafo qualificato.",
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", prefix="grapholab_referto_", delete=False)
    pdf.output(tmp.name)
    return tmp.name
