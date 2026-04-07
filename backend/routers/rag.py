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


class ModelSelect(BaseModel):
    model: str


class OpenAIKeyPayload(BaseModel):
    api_key: str


class DocInfo(BaseModel):
    filename: str
    chunks: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/model")
async def get_model(_: User = Depends(get_current_user)) -> dict:
    from core.rag import _rag_model
    return {"model": _rag_model}


@router.put("/model")
async def set_model(body: ModelSelect, _: User = Depends(get_current_user)) -> dict:
    from core.rag import set_rag_model
    msg = set_rag_model(body.model)
    return {"model": body.model, "detail": msg}


@router.get("/vlm-model")
async def get_vlm_model_endpoint(_: User = Depends(get_current_user)) -> dict:
    from core.rag import _vlm_model
    return {"vlm_model": _vlm_model}


@router.put("/vlm-model")
async def set_vlm_model_endpoint(body: ModelSelect, _: User = Depends(get_current_user)) -> dict:
    from core.rag import set_vlm_model
    msg = set_vlm_model(body.model)
    return {"vlm_model": body.model, "detail": msg}


@router.get("/ocr-model")
async def get_ocr_model_endpoint(_: User = Depends(get_current_user)) -> dict:
    from core.ocr import get_ocr_model
    return {"ocr_model": get_ocr_model()}


@router.put("/ocr-model")
async def set_ocr_model_endpoint(body: ModelSelect, _: User = Depends(get_current_user)) -> dict:
    from core.ocr import set_ocr_model
    msg = set_ocr_model(body.model)
    return {"ocr_model": body.model, "detail": msg}


@router.get("/status")
async def rag_status(_: User = Depends(get_current_user)) -> dict:
    from core.rag import check_ollama, ollama_list_models
    from core.providers import (
        openai_key_configured,
        OPENAI_LLM_MODELS,
        OPENAI_VLM_MODELS,
        OPENAI_EMBED_MODELS,
    )
    reachable = check_ollama()
    models = ollama_list_models() if reachable else []
    has_openai = openai_key_configured()
    return {
        "ollama_reachable": reachable,
        "models": models,
        "openai_available": has_openai,
        "openai_llm_models":   OPENAI_LLM_MODELS   if has_openai else [],
        "openai_vlm_models":   OPENAI_VLM_MODELS   if has_openai else [],
        "openai_embed_models": OPENAI_EMBED_MODELS  if has_openai else [],
    }


@router.get("/openai-key")
async def get_openai_key_status(_: User = Depends(get_current_user)) -> dict:
    from core.providers import openai_key_configured
    return {"configured": openai_key_configured()}


@router.put("/openai-key")
async def set_openai_key(
    body: OpenAIKeyPayload,
    _: User = Depends(get_current_user),
) -> dict:
    from core.providers import validate_openai_key, persist_openai_key, invalidate_openai_client
    try:
        valid = validate_openai_key(body.api_key)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    if not valid:
        raise HTTPException(status_code=400, detail="Chiave OpenAI non valida o non autorizzata")
    persist_openai_key(body.api_key)
    invalidate_openai_client()
    return {"ok": True}


@router.get("/embed-model")
async def get_embed_model_endpoint(_: User = Depends(get_current_user)) -> dict:
    from core.rag import get_embed_model
    return {"embed_model": get_embed_model()}


@router.put("/embed-model")
async def set_embed_model_endpoint(
    body: ModelSelect,
    _: User = Depends(get_current_user),
) -> dict:
    from core.rag import set_embed_model
    msg = set_embed_model(body.model)
    return {"embed_model": body.model, "detail": msg}


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
