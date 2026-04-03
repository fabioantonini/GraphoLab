"""
GraphoLab Backend — ENFSI Perizia Compliance Checker router.

Endpoints:
  POST /compliance/check   → upload perizia PDF, stream SSE compliance report
  GET  /compliance/status  → Ollama reachability check (same as RAG)
  POST /compliance/pdf     → generate PDF report from parsed compliance data
"""

from __future__ import annotations

import io
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from backend.auth.dependencies import get_current_user
from backend.models.user import User

router = APIRouter(prefix="/compliance", tags=["compliance"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class ReqBlockSchema(BaseModel):
    num: int
    name: str
    verdict: Literal["✅", "⚠️", "❌"] | None = None
    motivazione: str
    suggerimento: str | None = None


class CompliancePdfRequest(BaseModel):
    filename: str = "perizia"
    blocks: list[ReqBlockSchema]
    conformi: int
    parziali: int
    mancanti: int
    judgment: str


@router.get("/status")
async def compliance_status(_: User = Depends(get_current_user)) -> dict:
    from core.rag import check_ollama
    return {"ollama_reachable": check_ollama()}


@router.post("/check")
async def compliance_check(
    file: UploadFile = File(...),
    _: User = Depends(get_current_user),
) -> StreamingResponse:
    """
    Accept a PDF perizia, extract its text, then stream the ENFSI compliance
    analysis as Server-Sent Events.
    """
    from core.compliance import compliance_check_stream, extract_perizia_text

    # Save uploaded file to a temp location
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / Path(file.filename or "perizia.pdf").name
    tmp_path.write_bytes(await file.read())

    # Extract text synchronously (fast for a typical perizia)
    perizia_text = extract_perizia_text(tmp_path)

    async def _generate():
        if not perizia_text.strip():
            error_msg = "❌ Impossibile estrarre il testo dal PDF. Verifica che il file non sia corrotto o protetto da password."
            yield f"data: {json.dumps(error_msg)}\n\n"
            return

        for accumulated in compliance_check_stream(perizia_text):
            yield f"data: {json.dumps(accumulated)}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


# ── PDF generation ────────────────────────────────────────────────────────────

def _build_compliance_pdf(req: CompliancePdfRequest) -> bytes:
    from fpdf import FPDF

    VERDICT_LABEL = {"✅": "CONFORME", "⚠️": "PARZIALE", "❌": "MANCANTE"}
    VERDICT_RGB = {
        "✅": {"fill": (230, 247, 230), "bar": (76, 175, 80),  "text": (27, 109, 37)},
        "⚠️": {"fill": (255, 248, 225), "bar": (255, 193, 7),  "text": (130, 80, 0)},
        "❌": {"fill": (255, 235, 235), "bar": (229, 57, 53),  "text": (160, 30, 30)},
    }

    def _t(text: str) -> str:
        """Sanitize to latin-1, strip markdown markers."""
        if not text:
            return ""
        for src, dst in {
            "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
            "\u201c": '"', "\u201d": '"', "\u2026": "...",
            "✅": "[OK]", "⚠️": "[!]", "❌": "[X]", "💡": "",
        }.items():
            text = text.replace(src, dst)
        import re
        text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)
        text = re.sub(r"\*+", "", text)
        return text.encode("latin-1", errors="replace").decode("latin-1")

    class CompliancePDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(120, 120, 120)
            self.cell(0, 7, "GraphoLab  -  Report di Conformita ENFSI BPM", align="C")
            self.set_draw_color(200, 200, 200)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)

        def footer(self):
            self.set_y(-14)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 8, f"Pagina {self.page_no()}  -  Generato da GraphoLab il {datetime.now().strftime('%d/%m/%Y %H:%M')}", align="C")

    pdf = CompliancePDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    total = len(req.blocks)

    # ── Title block ───────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 10, "Report di Conformita ENFSI BPM", align="C")
    pdf.ln(7)
    # Horizontal rule under title
    pdf.set_draw_color(180, 180, 180)
    pdf.line(30, pdf.get_y(), 180, pdf.get_y())
    pdf.ln(4)
    # Filename: strip extension, truncate, smaller font
    short_name = _t(Path(req.filename).stem)
    if len(short_name) > 80:
        short_name = short_name[:77] + "..."
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(0, 5, short_name, align="C")
    pdf.ln(7)

    # ── Summary stats (3 col boxes) ───────────────────────────────────────────
    col_w = 58.0
    gap = 4.0
    start_x = (210 - (col_w * 3 + gap * 2)) / 2
    boxes = [
        ("CONFORMI", req.conformi, total, (230, 247, 230), (76, 175, 80), (27, 109, 37)),
        ("PARZIALI", req.parziali, total, (255, 248, 225), (255, 193, 7), (130, 80, 0)),
        ("MANCANTI", req.mancanti, total, (255, 235, 235), (229, 57, 53), (160, 30, 30)),
    ]
    box_y = pdf.get_y()
    for i, (label, count, tot, fill, border_c, text_c) in enumerate(boxes):
        bx = start_x + i * (col_w + gap)
        pdf.set_fill_color(*fill)
        pdf.set_draw_color(*border_c)
        pdf.rect(bx, box_y, col_w, 22, style="FD")
        pdf.set_xy(bx, box_y + 2)
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(*text_c)
        pdf.cell(col_w, 10, f"{count} / {tot}", align="C")
        pdf.set_xy(bx, box_y + 12)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(col_w, 7, label, align="C")
    pdf.ln(16)

    # ── Score bar ─────────────────────────────────────────────────────────────
    bar_x, bar_y, bar_w, bar_h = 10.0, pdf.get_y(), 190.0, 6.0
    segments = [
        (req.conformi,  (76, 175, 80)),
        (req.parziali,  (255, 193, 7)),
        (req.mancanti,  (229, 57, 53)),
    ]
    cx = bar_x
    for count, color in segments:
        if count > 0:
            sw = bar_w * count / total
            pdf.set_fill_color(*color)
            pdf.rect(cx, bar_y, sw, bar_h, style="F")
            cx += sw
    # Bar border
    pdf.set_draw_color(200, 200, 200)
    pdf.rect(bar_x, bar_y, bar_w, bar_h, style="D")
    pdf.ln(bar_h + 4)

    # ── Judgment banner ───────────────────────────────────────────────────────
    if req.judgment:
        jtext = _t(req.judgment)
        if "Buona" in req.judgment:
            jfill, jtext_c = (230, 247, 230), (27, 109, 37)
        elif "discreta" in req.judgment:
            jfill, jtext_c = (227, 242, 253), (13, 71, 161)
        elif "parziale" in req.judgment:
            jfill, jtext_c = (255, 248, 225), (130, 80, 0)
        else:
            jfill, jtext_c = (255, 235, 235), (160, 30, 30)
        pdf.set_fill_color(*jfill)
        pdf.set_draw_color(*jtext_c)
        pdf.set_text_color(*jtext_c)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 9, jtext, border=1, align="C", fill=True)
        pdf.ln(6)

    # ── Summary table ─────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, "Riepilogo requisiti", ln=True)
    pdf.ln(1)

    # Header row
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(210, 225, 240)
    pdf.set_text_color(30, 30, 30)
    pdf.set_draw_color(180, 180, 180)
    for txt, w in [("REQ", 18), ("Nome requisito", 130), ("Verdetto", 42)]:
        pdf.cell(w, 6, txt, border=1, fill=True)
    pdf.ln()

    for block in req.blocks:
        cfg = VERDICT_RGB.get(block.verdict or "", {"fill": (255,255,255), "bar": (180,180,180), "text": (80,80,80)})
        pdf.set_fill_color(*cfg["fill"])
        pdf.set_text_color(50, 50, 50)
        pdf.set_font("Helvetica", "", 8)
        row_y = pdf.get_y()
        pdf.cell(18, 5.5, f"REQ-{block.num:02d}", border=1, fill=True)
        pdf.cell(130, 5.5, _t(block.name)[:60], border=1, fill=True)
        pdf.set_text_color(*cfg["text"])
        pdf.set_font("Helvetica", "B", 8)
        label = VERDICT_LABEL.get(block.verdict or "", "-")
        pdf.cell(42, 5.5, label, border=1, fill=True, align="C")
        pdf.ln()
    # Note on summary page — adaptive position
    note_y = max(pdf.get_y() + 8, pdf.h - 42)
    pdf.set_y(note_y)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(140, 140, 140)
    pdf.cell(0, 5, "Analisi condotta secondo ENFSI Best Practice Manual for Forensic Examination of Handwriting (BPM-FHX-01 Ed.03, sez. 13.2)", align="C")

    # ── Detailed blocks ───────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 9, "Analisi dettagliata", ln=True)
    pdf.ln(2)

    for block in req.blocks:
        cfg = VERDICT_RGB.get(block.verdict or "", {"fill": (248,248,248), "bar": (180,180,180), "text": (80,80,80)})
        label = VERDICT_LABEL.get(block.verdict or "", "N/D")

        # Estimate block height to avoid page breaks inside
        lines_motiv = max(1, len(_t(block.motivazione)) // 95 + 1)
        lines_sugg  = max(1, len(_t(block.suggerimento or "")) // 95 + 1) if block.suggerimento else 0
        block_h = 8 + 5.5 * lines_motiv + (5.5 * lines_sugg + 6 if block.suggerimento else 0) + 6

        if pdf.get_y() + block_h > 272:
            pdf.add_page()

        bx, by = 10.0, pdf.get_y()

        # Colored left bar
        pdf.set_fill_color(*cfg["bar"])
        pdf.rect(bx, by, 3, block_h, style="F")

        # Card background
        pdf.set_fill_color(*cfg["fill"])
        pdf.set_draw_color(210, 210, 210)
        pdf.rect(bx + 3, by, 187, block_h, style="FD")

        # REQ number + name
        pdf.set_xy(bx + 5, by + 2)
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(30, 4, f"REQ-{block.num:02d}")

        pdf.set_xy(bx + 5, by + 6)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(140, 5, _t(block.name)[:70])

        # Verdict badge (right side)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*cfg["text"])
        pdf.set_xy(bx + 148, by + 5)
        pdf.cell(48, 6, label, align="R")

        # Motivazione
        pdf.set_xy(bx + 5, by + 12)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(60, 60, 60)
        pdf.multi_cell(182, 4.5, _t(block.motivazione), border=0)

        # Suggerimento
        if block.suggerimento:
            sy = pdf.get_y() + 1
            pdf.set_fill_color(255, 243, 205)
            pdf.set_draw_color(255, 193, 7)
            pdf.rect(bx + 5, sy, 182, 5.5 * lines_sugg + 3, style="FD")
            pdf.set_xy(bx + 7, sy + 1.5)
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(100, 60, 0)
            pdf.multi_cell(178, 4.5, "Suggerimento: " + _t(block.suggerimento), border=0)

        pdf.ln(2)

    # ── Closing block — adaptive position ─────────────────────────────────────
    # If content ends in the upper half of the page, place colophon right after
    # with a gap; otherwise pin it near the bottom to avoid excessive white space.
    colophon_y = max(pdf.get_y() + 10, pdf.h - 52)
    pdf.set_y(colophon_y)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(140, 140, 140)
    pdf.cell(0, 5, "Analisi condotta secondo ENFSI Best Practice Manual for Forensic Examination of Handwriting (BPM-FHX-01 Ed.03, sez. 13.2)", align="C")
    pdf.ln(4)
    pdf.cell(0, 5, f"Generato da GraphoLab il {datetime.now().strftime('%d/%m/%Y alle %H:%M')}  -  Documento riservato", align="C")

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


@router.post("/pdf")
async def compliance_pdf(
    body: CompliancePdfRequest,
    _: User = Depends(get_current_user),
) -> Response:
    """Generate a PDF report from a parsed compliance analysis."""
    import asyncio
    pdf_bytes = await asyncio.to_thread(_build_compliance_pdf, body)
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in Path(body.filename).stem)
    filename = f"compliance_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
