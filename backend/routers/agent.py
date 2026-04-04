"""
GraphoLab Backend — Agente Documentale router.

Endpoints:
  POST /agent/chat       → streaming agent response (SSE)
  GET  /agent/status     → Ollama reachability + agent model + tool list
  GET  /agent/prompts    → pre-formatted prompt templates
"""

from __future__ import annotations

import asyncio
import json
import queue
import tempfile
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import StreamingResponse

from backend.auth.dependencies import get_current_user
from backend.models.user import User

router = APIRouter(prefix="/agent", tags=["agent"])

_SENTINEL = object()  # signals end-of-stream from worker thread


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/status")
async def agent_status(_: User = Depends(get_current_user)) -> dict:
    """Return Ollama reachability, active agent model, and available tools."""
    from core.rag import check_ollama, ollama_list_models
    from core.agent import get_active_model, AGENT_TOOLS_NAMES

    reachable = check_ollama()
    models = ollama_list_models() if reachable else []
    return {
        "ollama_reachable": reachable,
        "models": models,
        "agent_model": get_active_model(),
        "tools": AGENT_TOOLS_NAMES,
    }


@router.get("/prompts")
async def get_prompts(_: User = Depends(get_current_user)) -> dict:
    """Return the list of pre-formatted prompt templates."""
    from core.agent import SUGGESTED_PROMPTS
    return {"prompts": SUGGESTED_PROMPTS}


@router.post("/chat")
async def chat(
    message: str = Form(...),
    history: str = Form(default="[]"),
    files: list[UploadFile] = File(default=[]),
    _: User = Depends(get_current_user),
) -> StreamingResponse:
    """Stream agent response as Server-Sent Events.

    Each SSE event carries a JSON-encoded string that is the *accumulated*
    assistant response so far (same pattern as /rag/chat and /compliance/check).

    Form fields:
      - message:  User text input
      - history:  JSON array of {"role": "user"|"assistant", "content": str}
      - files:    Optional uploaded files (images, PDFs)
    """
    # Parse history
    try:
        parsed_history: list[dict] = json.loads(history)
    except Exception:
        parsed_history = []

    # Save uploaded files to a temp directory; collect their paths
    tmp_dir = Path(tempfile.mkdtemp())
    file_paths: list[str] = []
    for upload in files:
        if upload and upload.filename:
            safe_name = Path(upload.filename).name
            dest = tmp_dir / safe_name
            dest.write_bytes(await upload.read())
            file_paths.append(str(dest))

    async def _generate():
        from core.agent import agent_stream

        stop_event = threading.Event()
        q: queue.Queue = queue.Queue()

        def _worker():
            try:
                for text in agent_stream(message, file_paths, parsed_history,
                                         stop_event=stop_event):
                    q.put(text)
            except Exception as e:
                q.put(f"❌ Errore: {e}")
            finally:
                q.put(_SENTINEL)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        loop = asyncio.get_event_loop()
        try:
            while True:
                item = await loop.run_in_executor(None, q.get)
                if item is _SENTINEL:
                    break
                yield f"data: {json.dumps(item)}\n\n"
        except (GeneratorExit, asyncio.CancelledError):
            # Client disconnected — signal the worker to stop
            stop_event.set()
            raise

    return StreamingResponse(_generate(), media_type="text/event-stream")
