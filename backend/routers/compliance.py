"""
GraphoLab Backend — ENFSI Perizia Compliance Checker router.

Endpoints:
  POST /compliance/check   → upload perizia PDF, stream SSE compliance report
  GET  /compliance/status  → Ollama reachability check (same as RAG)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile, status
from fastapi.responses import StreamingResponse

from backend.auth.dependencies import get_current_user
from backend.models.user import User

router = APIRouter(prefix="/compliance", tags=["compliance"])


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
