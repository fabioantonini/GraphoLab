"""
GraphoLab Backend — Analysis router.

All heavy AI work is delegated to core/. This router is a thin HTTP layer:
  1. Download the document image from MinIO
  2. Call the appropriate core/ function
  3. Persist the result text in the DB
  4. Return the result to the client

Endpoints:
  POST /analysis/htr                      → HTR transcription
  POST /analysis/signature-detection      → Conditional DETR signature detection
  POST /analysis/signature-verification   → SigNet verification
  POST /analysis/ner                      → Named Entity Recognition
  POST /analysis/writer                   → Writer identification
  POST /analysis/graphology               → Graphological analysis
  POST /analysis/pipeline                 → Full 7-step forensic pipeline
  POST /analysis/dating                   → Document dating
  GET  /analysis/project/{project_id}     → List analyses for a project
"""

from __future__ import annotations

import io
from pathlib import Path

import anyio
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.audit import log_event
from backend.auth.dependencies import get_current_user
from backend.config import settings
from backend.database import get_db
from backend.models.audit import AuditAction
from backend.models.project import Analysis, AnalysisType, Document, Project
from backend.models.user import Role, User
from backend.storage.minio_client import download_object, upload_fileobj

router = APIRouter(prefix="/analysis", tags=["analysis"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    project_id: int
    document_id: int


class SignatureVerifyRequest(BaseModel):
    project_id: int
    document_id: int           # document under examination
    reference_document_id: int  # known genuine signature document


class PipelineRequest(BaseModel):
    project_id: int
    document_id: int
    reference_document_id: int | None = None  # optional reference signature for step 6


class AnalysisOut(BaseModel):
    id: int
    analysis_type: AnalysisType
    result_text: str | None
    result_storage_key: str | None
    project_id: int
    document_id: int | None

    model_config = {"from_attributes": True}


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _load_image(doc: Document) -> np.ndarray:
    """Download a document from MinIO and return it as an RGB numpy array."""
    data = await download_object(doc.storage_key)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img)


async def _get_doc(doc_id: int, project_id: int, db: AsyncSession) -> Document:
    result = await db.execute(
        select(Document).where(Document.id == doc_id, Document.project_id == project_id)
    )
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Documento non trovato.")
    return doc


async def _check_project_access(project_id: int, db: AsyncSession, user: User) -> Project:
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Perizia non trovata.")
    if user.role != Role.admin and project.owner_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Accesso negato.")
    return project


async def _numpy_to_png(image: np.ndarray) -> bytes:
    """Convert a numpy RGB array to PNG bytes."""
    buf = io.BytesIO()
    Image.fromarray(image.astype("uint8")).save(buf, format="PNG")
    return buf.getvalue()


async def _save_analysis(
    db: AsyncSession,
    project_id: int,
    doc_id: int | None,
    analysis_type: AnalysisType,
    result_text: str,
    result_image: np.ndarray | None = None,
) -> Analysis:
    storage_key: str | None = None
    if result_image is not None:
        storage_key = f"projects/{project_id}/analyses/{analysis_type.value}_annotated.png"
        png_bytes = await _numpy_to_png(result_image)
        await upload_fileobj(storage_key, png_bytes, "image/png")

    analysis = Analysis(
        analysis_type=analysis_type,
        result_text=result_text,
        result_storage_key=storage_key,
        project_id=project_id,
        document_id=doc_id,
    )
    db.add(analysis)
    await db.flush()
    return analysis


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/htr", response_model=AnalysisOut, status_code=status.HTTP_201_CREATED)
async def run_htr(
    body: AnalysisRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Analysis:
    await _check_project_access(body.project_id, db, current_user)
    doc = await _get_doc(body.document_id, body.project_id, db)
    image = await _load_image(doc)

    from core.ocr import htr_transcribe
    text = htr_transcribe(image)

    result = await _save_analysis(db, body.project_id, doc.id, AnalysisType.htr, text)
    await log_event(db, current_user, AuditAction.analysis_run,
                    resource_type="analysis", resource_id=result.id, detail="htr")
    return result


@router.post("/signature-detection", response_model=AnalysisOut, status_code=status.HTTP_201_CREATED)
async def run_signature_detection(
    body: AnalysisRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Analysis:
    await _check_project_access(body.project_id, db, current_user)
    doc = await _get_doc(body.document_id, body.project_id, db)
    image = await _load_image(doc)

    from core.signature import detect_and_crop
    annotated, _, summary = detect_and_crop(image)

    result = await _save_analysis(
        db, body.project_id, doc.id, AnalysisType.signature_detection, summary, annotated
    )
    await log_event(db, current_user, AuditAction.analysis_run,
                    resource_type="analysis", resource_id=result.id, detail="signature_detection")
    return result


@router.post("/signature-verification", response_model=AnalysisOut, status_code=status.HTTP_201_CREATED)
async def run_signature_verification(
    body: SignatureVerifyRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Analysis:
    await _check_project_access(body.project_id, db, current_user)
    doc = await _get_doc(body.document_id, body.project_id, db)
    ref_doc = await _get_doc(body.reference_document_id, body.project_id, db)

    image = await _load_image(doc)
    ref_image = await _load_image(ref_doc)

    from core.signature import detect_and_crop, sig_verify
    _, query_crop, _ = detect_and_crop(image)
    query = query_crop if query_crop is not None else image

    report, _ = sig_verify(ref_image, None, query, settings.signet_weights)

    result = await _save_analysis(
        db, body.project_id, doc.id, AnalysisType.signature_verification, report
    )
    await log_event(db, current_user, AuditAction.analysis_run,
                    resource_type="analysis", resource_id=result.id, detail="signature_verification")
    return result


@router.post("/ner", response_model=AnalysisOut, status_code=status.HTTP_201_CREATED)
async def run_ner(
    body: AnalysisRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Analysis:
    await _check_project_access(body.project_id, db, current_user)
    doc = await _get_doc(body.document_id, body.project_id, db)
    image = await _load_image(doc)

    from core.ocr import htr_transcribe
    from core.ner import ner_extract
    text = htr_transcribe(image)
    _, ner_summary = ner_extract(text)

    result = await _save_analysis(db, body.project_id, doc.id, AnalysisType.ner, ner_summary)
    await log_event(db, current_user, AuditAction.analysis_run,
                    resource_type="analysis", resource_id=result.id, detail="ner")
    return result


@router.post("/writer", response_model=AnalysisOut, status_code=status.HTTP_201_CREATED)
async def run_writer_identification(
    body: AnalysisRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Analysis:
    await _check_project_access(body.project_id, db, current_user)
    doc = await _get_doc(body.document_id, body.project_id, db)
    image = await _load_image(doc)

    from core.writer import writer_identify
    report, _ = writer_identify(image, settings.writer_samples_dir)

    result = await _save_analysis(
        db, body.project_id, doc.id, AnalysisType.writer_identification, report
    )
    await log_event(db, current_user, AuditAction.analysis_run,
                    resource_type="analysis", resource_id=result.id, detail="writer_identification")
    return result


@router.post("/graphology", response_model=AnalysisOut, status_code=status.HTTP_201_CREATED)
async def run_graphology(
    body: AnalysisRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Analysis:
    await _check_project_access(body.project_id, db, current_user)
    doc = await _get_doc(body.document_id, body.project_id, db)
    image = await _load_image(doc)

    from core.graphology import grapho_analyse
    report, annotated = grapho_analyse(image)

    result = await _save_analysis(
        db, body.project_id, doc.id, AnalysisType.graphology, report, annotated
    )
    await log_event(db, current_user, AuditAction.analysis_run,
                    resource_type="analysis", resource_id=result.id, detail="graphology")
    return result


@router.post("/dating", response_model=AnalysisOut, status_code=status.HTTP_201_CREATED)
async def run_dating(
    body: AnalysisRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Analysis:
    await _check_project_access(body.project_id, db, current_user)
    doc = await _get_doc(body.document_id, body.project_id, db)
    image = await _load_image(doc)

    from core.ocr import htr_transcribe
    from core.dating import extract_dates
    text = htr_transcribe(image)
    dates = extract_dates(text)
    if dates:
        dating_result = "\n".join(f"- {raw} → {dt.strftime('%Y-%m-%d')}" for raw, dt in dates)
    else:
        dating_result = "Nessuna data rilevata nel documento."

    saved = await _save_analysis(db, body.project_id, doc.id, AnalysisType.dating, dating_result)
    await log_event(db, current_user, AuditAction.analysis_run,
                    resource_type="analysis", resource_id=saved.id, detail="dating")
    return saved


@router.post("/pipeline", response_model=AnalysisOut, status_code=status.HTTP_201_CREATED)
async def run_pipeline(
    body: PipelineRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Analysis:
    """Run the full 7-step forensic pipeline (blocking — may take 30–120 s)."""
    await _check_project_access(body.project_id, db, current_user)
    doc = await _get_doc(body.document_id, body.project_id, db)
    image = await _load_image(doc)

    ref_sig = None
    if body.reference_document_id:
        ref_doc = await _get_doc(body.reference_document_id, body.project_id, db)
        ref_sig = await _load_image(ref_doc)

    from core.pipeline import run_pipeline_steps
    results = None
    for results in run_pipeline_steps(
        image, ref_sig, settings.signet_weights, settings.writer_samples_dir
    ):
        pass  # consume generator to completion

    if not results:
        return await _save_analysis(db, body.project_id, doc.id, AnalysisType.pipeline, "Pipeline non completata.")

    # Save step images to storage and embed placeholder markers in the report
    sig_key: str | None = None
    grapho_key: str | None = None
    if results.sig_detect_image is not None:
        sig_key = f"projects/{body.project_id}/pipeline/step1_signature.png"
        await upload_fileobj(sig_key, await _numpy_to_png(results.sig_detect_image), "image/png")
    if results.grapho_image is not None:
        grapho_key = f"projects/{body.project_id}/pipeline/step5_graphology.png"
        await upload_fileobj(grapho_key, await _numpy_to_png(results.grapho_image), "image/png")

    # Build report with image markers that the frontend can replace
    report = results.final_report
    if sig_key:
        report = report.replace(
            "### Step 1 — Rilevamento Firma\n",
            f"### Step 1 — Rilevamento Firma\n\n![sig_detect](__img_sig__)\n\n",
        )
    if grapho_key:
        report = report.replace(
            "### Step 5 — Caratteristiche Grafologiche\n",
            f"### Step 5 — Caratteristiche Grafologiche\n\n![grapho](__img_grapho__)\n\n",
        )

    # Pass None for result_image so no result_storage_key is set — images are embedded inline in the report
    analysis = await _save_analysis(db, body.project_id, doc.id, AnalysisType.pipeline, report, None)

    # Store grapho image key as secondary (overwrite storage key with sig, store grapho separately)
    # We embed both keys in the report text so the frontend can fetch them by analysis id
    if grapho_key:
        analysis.result_text = analysis.result_text.replace("__img_grapho__", f"/api/analysis/{analysis.id}/image/grapho") if analysis.result_text else analysis.result_text
    if sig_key:
        analysis.result_text = analysis.result_text.replace("__img_sig__", f"/api/analysis/{analysis.id}/image/sig") if analysis.result_text else analysis.result_text
    await db.flush()

    await log_event(db, current_user, AuditAction.analysis_run,
                    resource_type="analysis", resource_id=analysis.id, detail="pipeline")
    return analysis


@router.get("/project/{project_id}", response_model=list[AnalysisOut])
async def list_analyses(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[Analysis]:
    await _check_project_access(project_id, db, current_user)
    result = await db.execute(
        select(Analysis)
        .where(Analysis.project_id == project_id)
        .order_by(Analysis.created_at.desc())
    )
    return result.scalars().all()


@router.delete("/project/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def clear_analyses(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    await _check_project_access(project_id, db, current_user)
    await db.execute(delete(Analysis).where(Analysis.project_id == project_id))
    await log_event(db, current_user, AuditAction.analysis_clear,
                    resource_type="project", resource_id=project_id)
    await db.commit()


@router.get("/{analysis_id}/image")
async def get_analysis_image(
    analysis_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
    analysis = result.scalar_one_or_none()
    if analysis is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analisi non trovata.")
    await _check_project_access(analysis.project_id, db, current_user)
    if not analysis.result_storage_key:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Nessuna immagine disponibile.")
    data = await download_object(analysis.result_storage_key)
    return StreamingResponse(io.BytesIO(data), media_type="image/png")


@router.get("/{analysis_id}/image/{slot}")
async def get_analysis_image_slot(
    analysis_id: int,
    slot: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
    analysis = result.scalar_one_or_none()
    if analysis is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analisi non trovata.")
    await _check_project_access(analysis.project_id, db, current_user)
    slot_map = {
        "sig": f"projects/{analysis.project_id}/pipeline/step1_signature.png",
        "grapho": f"projects/{analysis.project_id}/pipeline/step5_graphology.png",
    }
    key = slot_map.get(slot)
    if not key:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Slot non valido.")
    data = await download_object(key)
    return StreamingResponse(io.BytesIO(data), media_type="image/png")


@router.get("/{analysis_id}/pdf")
async def get_analysis_pdf(
    analysis_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
    analysis = result.scalar_one_or_none()
    if analysis is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analisi non trovata.")
    await _check_project_access(analysis.project_id, db, current_user)
    if not analysis.result_text:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Nessun testo disponibile.")

    await log_event(db, current_user, AuditAction.pdf_download,
                    resource_type="analysis", resource_id=analysis_id)
    pdf_bytes = await anyio.to_thread.run_sync(lambda: _generate_pdf(analysis))
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=referto_{analysis_id}.pdf"},
    )


def _generate_pdf(analysis: Analysis) -> bytes:
    import re
    from datetime import datetime
    from fpdf import FPDF
    from PIL import Image as PILImage
    from backend.storage.minio_client import _download_sync

    # Storage keys for pipeline images
    _slot_map = {
        "sig": f"projects/{analysis.project_id}/pipeline/step1_signature.png",
        "grapho": f"projects/{analysis.project_id}/pipeline/step5_graphology.png",
    }

    def _t(text: str) -> str:
        """Sanitize text to latin-1 and strip basic markdown formatting."""
        if not text:
            return ""
        replacements = {
            "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
            "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u2022": "*",
            "\u2713": "v", "\u2714": "v", "\u2718": "x",
            "\U0001f947": "1.", "\U0001f948": "2.", "\U0001f949": "3.",
            "\u26a0\ufe0f": "(!)", "\u26a0": "(!)",
            "\U0001f50d": "", "\U0001f5d1": "",
        }
        for src, dst in replacements.items():
            text = text.replace(src, dst)
        text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)
        text = re.sub(r"\*+", "", text)  # remove orphaned asterisks (e.g. from LLM **label**: pattern)
        text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)
        return text.encode("latin-1", errors="replace").decode("latin-1")

    def _embed_image(pdf: FPDF, img_bytes: bytes, max_w: float = 170.0) -> None:
        try:
            buf_in = io.BytesIO(img_bytes)
            img = PILImage.open(buf_in).convert("RGB")
            w, h = img.size
            ratio = min(max_w / w, 100.0 / h)
            disp_w, disp_h = w * ratio, h * ratio
            buf_out = io.BytesIO()
            img.save(buf_out, format="JPEG", quality=85)
            buf_out.seek(0)
            x = (210 - disp_w) / 2
            pdf.image(buf_out, x=x, w=disp_w, h=disp_h)
            pdf.ln(4)
        except Exception:
            pass

    def _render_table(pdf: FPDF, rows: list[list[str]]) -> None:
        if not rows:
            return
        n_cols = max(len(r) for r in rows)
        if n_cols == 0:
            return
        col_w = 190.0 / n_cols
        for row_idx, row in enumerate(rows):
            is_header = row_idx == 0
            if is_header:
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_fill_color(210, 225, 240)
            else:
                pdf.set_font("Helvetica", "", 9)
                if row_idx % 2 == 0:
                    pdf.set_fill_color(245, 248, 252)
                else:
                    pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(30, 30, 30)
            for c_idx in range(n_cols):
                cell_text = _t(row[c_idx]) if c_idx < len(row) else ""
                pdf.cell(col_w, 6, cell_text, border=1, fill=True)
            pdf.ln()
        pdf.ln(4)

    class ReportPDF(FPDF):
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

    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # Title block — clearly separated from header line
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 14, "Referto Forense Integrato", align="C")
    pdf.ln(16)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Data generazione: {datetime.now().strftime('%d/%m/%Y %H:%M')}", align="C")
    pdf.ln(12)

    raw = analysis.result_text or ""
    lines = raw.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Empty line or horizontal rule
        if not stripped or stripped == "---":
            pdf.ln(2)
            i += 1
            continue

        # H2 heading — skip the document title (already rendered at top)
        if stripped.startswith("## "):
            if "Referto Forense Integrato" not in stripped:
                pdf.set_font("Helvetica", "B", 14)
                pdf.set_text_color(30, 30, 30)
                pdf.cell(0, 10, _t(stripped[3:]))
                pdf.ln(8)
            i += 1
            continue

        # H3 section title bar
        if stripped.startswith("### "):
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(255, 255, 255)
            pdf.set_fill_color(50, 80, 120)
            pdf.cell(0, 8, _t(f"  {stripped[4:]}"), fill=True)
            pdf.ln(10)
            pdf.set_text_color(30, 30, 30)
            i += 1
            continue

        # Image tag — download from storage and embed
        img_match = re.match(r"!\[.*?\]\(/api/analysis/\d+/image/(\w+)\)", stripped)
        if img_match:
            slot = img_match.group(1)
            key = _slot_map.get(slot)
            if key:
                try:
                    img_bytes = _download_sync(key)
                    _embed_image(pdf, img_bytes)
                except Exception:
                    pass
            i += 1
            continue

        # Table: collect all consecutive pipe-delimited lines
        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines: list[str] = []
            while i < len(lines):
                sl = lines[i].strip()
                if sl.startswith("|") and sl.endswith("|"):
                    table_lines.append(sl)
                    i += 1
                else:
                    break
            # Drop separator rows (|---|---|) and parse cells
            parsed_rows: list[list[str]] = []
            for row_str in table_lines:
                if re.match(r"^\|[-| :]+\|$", row_str):
                    continue
                cells = [c.strip() for c in row_str.strip("|").split("|")]
                parsed_rows.append(cells)
            _render_table(pdf, parsed_rows)
            continue

        # Regular text
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 5, _t(stripped))
        pdf.ln(1)
        i += 1

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()
