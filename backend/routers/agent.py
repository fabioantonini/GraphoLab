"""
GraphoLab Backend — Agente Documentale router.

Endpoints:
  POST /agent/chat                              → streaming agent response (SSE)
  GET  /agent/status                            → Ollama reachability + agent model + tool list
  GET  /agent/prompts                           → pre-formatted prompt templates

  GET    /agent/projects/                       → list agent projects
  POST   /agent/projects/                       → create agent project
  DELETE /agent/projects/{project_id}           → delete agent project

  GET    /agent/projects/{project_id}/chats     → list chats in project
  POST   /agent/projects/{project_id}/chats     → create new chat
  GET    /agent/chats/{chat_id}                 → get chat with messages
  DELETE /agent/chats/{chat_id}                 → delete chat

  GET    /agent/projects/{project_id}/documents → list project documents
  DELETE /agent/projects/{project_id}/documents/{doc_id} → delete document
"""

from __future__ import annotations

import asyncio
import json
import queue
import tempfile
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.audit import log_event
from backend.auth.dependencies import get_current_user
from backend.database import get_db
from backend.models.audit import AuditAction
from backend.models.project import AgentChat, AgentMessage, Document, Project, ProjectStatus
from backend.models.user import Role, User
from backend.storage.minio_client import delete_object, download_object, upload_fileobj

router = APIRouter(prefix="/agent", tags=["agent"])

_SENTINEL = object()  # signals end-of-stream from worker thread


# ── Schemas ───────────────────────────────────────────────────────────────────

class ProjectCreate(BaseModel):
    title: str


class ProjectOut(BaseModel):
    id: int
    title: str
    owner_id: int
    chat_count: int = 0

    model_config = {"from_attributes": True}


class ChatOut(BaseModel):
    id: int
    project_id: int
    title: str
    created_at: str

    model_config = {"from_attributes": True}


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    file_ids: list[int] = []
    created_at: str

    model_config = {"from_attributes": True}


class ChatDetailOut(BaseModel):
    id: int
    project_id: int
    title: str
    created_at: str
    messages: list[MessageOut] = []

    model_config = {"from_attributes": True}


class DocumentOut(BaseModel):
    id: int
    filename: str
    content_type: str
    size_bytes: int
    storage_key: str

    model_config = {"from_attributes": True}


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _get_project_or_404(project_id: int, db: AsyncSession, current_user: User) -> Project:
    result = await db.execute(
        select(Project)
        .options(selectinload(Project.documents), selectinload(Project.agent_chats))
        .where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=404, detail="Progetto non trovato.")
    if current_user.role != Role.admin and project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Accesso negato.")
    return project


async def _get_chat_or_404(chat_id: int, db: AsyncSession, current_user: User) -> AgentChat:
    result = await db.execute(
        select(AgentChat)
        .options(selectinload(AgentChat.messages), selectinload(AgentChat.project))
        .where(AgentChat.id == chat_id)
    )
    chat = result.scalar_one_or_none()
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat non trovata.")
    if current_user.role != Role.admin and chat.project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Accesso negato.")
    return chat


def _serialize_chat(chat: AgentChat) -> dict:
    return {
        "id": chat.id,
        "project_id": chat.project_id,
        "title": chat.title,
        "created_at": chat.created_at.isoformat(),
    }


def _serialize_message(msg: AgentMessage) -> dict:
    return {
        "id": msg.id,
        "role": msg.role,
        "content": msg.content,
        "file_ids": json.loads(msg.file_ids) if msg.file_ids else [],
        "created_at": msg.created_at.isoformat(),
    }


# ── Status / Prompts ──────────────────────────────────────────────────────────

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


@router.get("/temp-image/{filename}")
async def serve_temp_image(
    filename: str,
    _: User = Depends(get_current_user),
) -> StreamingResponse:
    """Serve an image file saved in %TEMP%/gl/ by agent tools."""
    import mimetypes
    from fastapi.responses import FileResponse

    gl_dir = Path(tempfile.gettempdir()) / "gl"
    file_path = (gl_dir / filename).resolve()
    if not str(file_path).startswith(str(gl_dir.resolve())):
        raise HTTPException(status_code=403, detail="Access denied")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    mime, _ = mimetypes.guess_type(str(file_path))
    return FileResponse(str(file_path), media_type=mime or "application/octet-stream")


@router.get("/prompts")
async def get_prompts(_: User = Depends(get_current_user)) -> dict:
    """Return the list of pre-formatted prompt templates."""
    from core.agent import SUGGESTED_PROMPTS
    return {"prompts": SUGGESTED_PROMPTS}


# ── Agent Projects ────────────────────────────────────────────────────────────

@router.get("/projects/")
async def list_agent_projects(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[dict]:
    """List projects that have at least one agent chat, or all projects owned by user."""
    if current_user.role == Role.admin:
        q = select(Project).options(selectinload(Project.agent_chats)).order_by(Project.id.desc())
    else:
        q = (
            select(Project)
            .options(selectinload(Project.agent_chats))
            .where(Project.owner_id == current_user.id)
            .order_by(Project.id.desc())
        )
    result = await db.execute(q)
    projects = result.scalars().all()
    return [
        {
            "id": p.id,
            "title": p.title,
            "owner_id": p.owner_id,
            "chat_count": len(p.agent_chats),
        }
        for p in projects
    ]


@router.post("/projects/", status_code=status.HTTP_201_CREATED)
async def create_agent_project(
    body: ProjectCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    project = Project(
        title=body.title,
        owner_id=current_user.id,
        status=ProjectStatus.in_progress,
    )
    db.add(project)
    await db.flush()
    await log_event(db, current_user, AuditAction.project_create,
                    resource_type="project", resource_id=project.id, detail=body.title)
    return {"id": project.id, "title": project.title, "owner_id": project.owner_id, "chat_count": 0}


@router.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent_project(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    project = await _get_project_or_404(project_id, db, current_user)
    await log_event(db, current_user, AuditAction.project_delete,
                    resource_type="project", resource_id=project_id, detail=project.title)
    for doc in project.documents:
        await delete_object(doc.storage_key)
    await db.delete(project)


# ── Agent Chats ───────────────────────────────────────────────────────────────

@router.get("/projects/{project_id}/chats")
async def list_chats(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[dict]:
    project = await _get_project_or_404(project_id, db, current_user)
    chats = sorted(project.agent_chats, key=lambda c: c.created_at, reverse=True)
    return [_serialize_chat(c) for c in chats]


@router.post("/projects/{project_id}/chats", status_code=status.HTTP_201_CREATED)
async def create_chat(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    await _get_project_or_404(project_id, db, current_user)
    chat = AgentChat(project_id=project_id, title="Nuova chat")
    db.add(chat)
    await db.flush()
    return _serialize_chat(chat)


@router.get("/chats/{chat_id}")
async def get_chat(
    chat_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    chat = await _get_chat_or_404(chat_id, db, current_user)
    return {
        **_serialize_chat(chat),
        "messages": [_serialize_message(m) for m in chat.messages],
    }


@router.delete("/chats/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(
    chat_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    chat = await _get_chat_or_404(chat_id, db, current_user)
    await db.delete(chat)


# ── Project Documents ─────────────────────────────────────────────────────────

@router.get("/projects/{project_id}/documents")
async def list_project_documents(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[dict]:
    project = await _get_project_or_404(project_id, db, current_user)
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "content_type": d.content_type,
            "size_bytes": d.size_bytes,
            "storage_key": d.storage_key,
        }
        for d in project.documents
    ]


@router.delete("/projects/{project_id}/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project_document(
    project_id: int,
    doc_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    await _get_project_or_404(project_id, db, current_user)
    result = await db.execute(
        select(Document).where(Document.id == doc_id, Document.project_id == project_id)
    )
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail="Documento non trovato.")
    await log_event(db, current_user, AuditAction.document_delete,
                    resource_type="document", resource_id=doc_id, detail=doc.filename)
    await delete_object(doc.storage_key)
    await db.delete(doc)


# ── Chat (streaming) ──────────────────────────────────────────────────────────

@router.post("/chat")
async def chat(
    message: str = Form(...),
    history: str = Form(default="[]"),
    files: list[UploadFile] = File(default=[]),
    project_id: int | None = Form(default=None),
    chat_id: int | None = Form(default=None),
    reused_doc_ids: str = Form(default="[]"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Stream agent response as Server-Sent Events.

    Optional form fields for persistent mode:
      - project_id:    Save files as Documents in this project
      - chat_id:       Persist messages to this AgentChat
      - reused_doc_ids: JSON list of existing Document.id to include (no re-upload)
    """
    try:
        parsed_history: list[dict] = json.loads(history)
    except Exception:
        parsed_history = []

    try:
        reused_ids: list[int] = json.loads(reused_doc_ids)
    except Exception:
        reused_ids = []

    # Save uploaded files — to project (MinIO) if project_id given, else tmp
    tmp_dir = Path(tempfile.mkdtemp())
    file_paths: list[str] = []
    new_doc_ids: list[int] = []

    for upload in files:
        if not upload or not upload.filename:
            continue
        content = await upload.read()
        safe_name = Path(upload.filename).name

        if project_id is not None:
            storage_key = f"projects/{project_id}/agent/{safe_name}"
            await upload_fileobj(storage_key, content, upload.content_type or "application/octet-stream")
            doc = Document(
                filename=safe_name,
                content_type=upload.content_type or "application/octet-stream",
                storage_key=storage_key,
                size_bytes=len(content),
                project_id=project_id,
            )
            db.add(doc)
            await db.flush()
            new_doc_ids.append(doc.id)
            await log_event(db, current_user, AuditAction.document_upload,
                            resource_type="document", resource_id=doc.id, detail=safe_name)
            # Write to tmp for agent processing
            dest = tmp_dir / safe_name
            dest.write_bytes(content)
            file_paths.append(str(dest))
        else:
            dest = tmp_dir / safe_name
            dest.write_bytes(content)
            file_paths.append(str(dest))

    # Resolve reused documents from MinIO
    if reused_ids:
        result = await db.execute(select(Document).where(Document.id.in_(reused_ids)))
        reused_docs = result.scalars().all()
        for doc in reused_docs:
            try:
                data = await download_object(doc.storage_key)
                dest = tmp_dir / doc.filename
                dest.write_bytes(data)
                file_paths.append(str(dest))
            except Exception:
                pass  # skip if file unavailable

    all_file_ids = reused_ids + new_doc_ids

    # Persist user message
    if chat_id is not None:
        result = await db.execute(select(AgentChat).where(AgentChat.id == chat_id))
        chat_obj = result.scalar_one_or_none()
        if chat_obj:
            user_msg = AgentMessage(
                chat_id=chat_id,
                role="user",
                content=message,
                file_ids=json.dumps(all_file_ids) if all_file_ids else None,
            )
            db.add(user_msg)
            # Auto-title chat from first user message
            if chat_obj.title == "Nuova chat":
                chat_obj.title = message[:60]
            await db.flush()

    # Build project context to inject into the agent system prompt
    project_context: str | None = None
    if project_id is not None:
        ctx_lines: list[str] = ["--- CONTESTO PROGETTO ---"]
        # Documents in the project
        doc_result = await db.execute(
            select(Document).where(Document.project_id == project_id)
        )
        all_docs = doc_result.scalars().all()
        if all_docs:
            ctx_lines.append("Documenti disponibili nel progetto:")
            for d in all_docs:
                ctx_lines.append(f"  - {d.filename} (id={d.id})")
        # Previous chats summary (titles only — keep context short)
        chat_result = await db.execute(
            select(AgentChat)
            .options(selectinload(AgentChat.messages))
            .where(AgentChat.project_id == project_id)
        )
        all_chats = chat_result.scalars().all()
        past_chats = [c for c in all_chats if c.id != chat_id and c.title != "Nuova chat"]
        if past_chats:
            ctx_lines.append("Chat precedenti nel progetto:")
            for c in past_chats[-5:]:  # last 5 chats max
                ctx_lines.append(f"  - {c.title}")
        ctx_lines.append("--- FINE CONTESTO ---")
        project_context = "\n".join(ctx_lines)

    # Commit DB changes before streaming (so files/messages are persisted even if client disconnects)
    await db.commit()

    async def _generate():
        from core.agent import agent_stream

        stop_event = threading.Event()
        q: queue.Queue = queue.Queue()
        final_text: list[str] = [""]

        def _worker():
            try:
                for text in agent_stream(message, file_paths, parsed_history,
                                         stop_event=stop_event,
                                         project_context=project_context):
                    final_text[0] = text
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
            stop_event.set()
            raise
        finally:
            # Persist assistant message after streaming completes
            if chat_id is not None and final_text[0]:
                from backend.database import AsyncSessionLocal
                async with AsyncSessionLocal() as save_db:
                    asst_msg = AgentMessage(
                        chat_id=chat_id,
                        role="assistant",
                        content=final_text[0],
                    )
                    save_db.add(asst_msg)
                    await save_db.commit()

    return StreamingResponse(_generate(), media_type="text/event-stream")
