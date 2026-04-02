"""
GraphoLab Backend — RAG / Consulente Forense IA router.

Endpoints:
  POST /rag/chat           → streaming chat (SSE)
  GET  /rag/docs           → list loaded documents
  POST /rag/docs           → upload and index a new document
  DELETE /rag/docs/{name}  → remove a document from the index
  GET  /rag/status         → Ollama reachability check
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.auth.dependencies import get_current_user
from backend.config import settings
from backend.models.user import User

router = APIRouter(prefix="/rag", tags=["rag"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []  # [{"role": "user"|"assistant", "content": str}, ...]


class DocInfo(BaseModel):
    filename: str
    chunks: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/status")
async def rag_status(_: User = Depends(get_current_user)) -> dict:
    from core.rag import check_ollama, ollama_list_models
    reachable = check_ollama()
    models = ollama_list_models() if reachable else []
    return {"ollama_reachable": reachable, "models": models}


@router.get("/docs", response_model=list[DocInfo])
async def list_docs(_: User = Depends(get_current_user)) -> list[DocInfo]:
    from core.rag import rag_doc_list
    docs = rag_doc_list()
    return [DocInfo(filename=row[0], chunks=row[1]) for row in docs]


@router.post("/docs", status_code=status.HTTP_201_CREATED)
async def add_doc(
    file: UploadFile = File(...),
    _: User = Depends(get_current_user),
) -> dict:
    import tempfile
    from pathlib import Path
    from core.rag import rag_add_docs

    original_name = Path(file.filename).name
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / original_name
    tmp_path.write_bytes(await file.read())

    # rag_add_docs expects a list of file-like objects with a .name attribute
    class _FileLike:
        name = str(tmp_path)

    result = rag_add_docs([_FileLike()], settings.rag_cache_dir)
    return {"detail": result}


@router.delete("/docs/{filename}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_doc(
    filename: str,
    _: User = Depends(get_current_user),
) -> None:
    from core.rag import rag_remove_doc
    rag_remove_doc(filename, settings.rag_cache_dir)


@router.post("/chat")
async def chat(
    body: ChatRequest,
    _: User = Depends(get_current_user),
) -> StreamingResponse:
    """Server-Sent Events stream of the RAG response."""
    import json as _json
    from core.rag import rag_chat_stream

    async def _generate():
        # Mirror the Gradio wrapper: combine partial + sources_footer in one update
        for partial_text, sources_footer in rag_chat_stream(body.message, body.history):
            content = partial_text + (sources_footer or "")
            yield f"data: {_json.dumps(content)}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")
