"""
GraphoLab Backend — Analysis router.

All heavy AI work is delegated to core/. This router is a thin HTTP layer:
  1. Download the document image from MinIO
  2. Call the appropriate core/ function
  3. Persist the result text in the DB
  4. Return the result to the client

Endpoints:
  POST /analysis/htr                      → HTR transcription
  POST /analysis/signature-detection      → YOLO signature detection
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

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth.dependencies import get_current_user
from backend.config import settings
from backend.database import get_db
from backend.models.project import Analysis, AnalysisType, Document, Project
from backend.models.user import Role, User
from backend.storage.minio_client import download_object

router = APIRouter(prefix="/analysis", tags=["analysis"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    project_id: int
    document_id: int


class SignatureVerifyRequest(BaseModel):
    project_id: int
    document_id: int           # document under examination
    reference_document_id: int  # known genuine signature document


class AnalysisOut(BaseModel):
    id: int
    analysis_type: AnalysisType
    result_text: str | None
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


async def _save_analysis(
    db: AsyncSession,
    project_id: int,
    doc_id: int | None,
    analysis_type: AnalysisType,
    result_text: str,
) -> Analysis:
    analysis = Analysis(
        analysis_type=analysis_type,
        result_text=result_text,
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

    return await _save_analysis(db, body.project_id, doc.id, AnalysisType.htr, text)


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
    _, _, summary = detect_and_crop(image)

    return await _save_analysis(
        db, body.project_id, doc.id, AnalysisType.signature_detection, summary
    )


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

    return await _save_analysis(
        db, body.project_id, doc.id, AnalysisType.signature_verification, report
    )


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

    return await _save_analysis(db, body.project_id, doc.id, AnalysisType.ner, ner_summary)


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

    return await _save_analysis(
        db, body.project_id, doc.id, AnalysisType.writer_identification, report
    )


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
    report, _ = grapho_analyse(image)

    return await _save_analysis(
        db, body.project_id, doc.id, AnalysisType.graphology, report
    )


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
        result = "\n".join(f"- {raw} → {dt.strftime('%Y-%m-%d')}" for raw, dt in dates)
    else:
        result = "Nessuna data rilevata nel documento."

    return await _save_analysis(db, body.project_id, doc.id, AnalysisType.dating, result)


@router.post("/pipeline", response_model=AnalysisOut, status_code=status.HTTP_201_CREATED)
async def run_pipeline(
    body: AnalysisRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Analysis:
    """Run the full 7-step forensic pipeline (blocking — may take 30–120 s)."""
    await _check_project_access(body.project_id, db, current_user)
    doc = await _get_doc(body.document_id, body.project_id, db)
    image = await _load_image(doc)

    from core.pipeline import run_pipeline_steps
    results = None
    for results in run_pipeline_steps(
        image, None, settings.signet_weights, settings.writer_samples_dir
    ):
        pass  # consume generator to completion

    report = results.final_report if results else "Pipeline non completata."
    return await _save_analysis(db, body.project_id, doc.id, AnalysisType.pipeline, report)


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
    await db.commit()
