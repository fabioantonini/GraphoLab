"""
GraphoLab Backend — Projects router.

A project (= "Perizia") groups documents and analyses for one case.

Endpoints:
  GET    /projects/              → list own projects
  POST   /projects/              → create project
  GET    /projects/{id}          → get project detail
  PUT    /projects/{id}          → update project
  DELETE /projects/{id}          → delete project
  POST   /projects/{id}/documents → upload document to project
  GET    /projects/{id}/documents → list project documents
  DELETE /projects/{id}/documents/{doc_id} → remove document
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.audit import log_event
from backend.auth.dependencies import get_current_user
from backend.database import get_db
from backend.models.audit import AuditAction
from backend.models.project import Document, Project, ProjectStatus
from backend.models.user import Role, User
from backend.storage.minio_client import delete_object, upload_fileobj

router = APIRouter(prefix="/projects", tags=["projects"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class ProjectOut(BaseModel):
    id: int
    title: str
    description: str | None
    status: ProjectStatus
    owner_id: int
    document_count: int = 0

    model_config = {"from_attributes": True}


class ProjectCreate(BaseModel):
    title: str
    description: str | None = None


class ProjectUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    status: ProjectStatus | None = None


class DocumentOut(BaseModel):
    id: int
    filename: str
    content_type: str
    size_bytes: int
    storage_key: str

    model_config = {"from_attributes": True}


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _get_project_or_404(
    project_id: int,
    db: AsyncSession,
    current_user: User,
) -> Project:
    result = await db.execute(
        select(Project)
        .options(selectinload(Project.documents))
        .where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Perizia non trovata.")
    # Admins can access all projects; examiners/viewers only their own
    if current_user.role != Role.admin and project.owner_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Accesso negato.")
    return project


# ── Project CRUD ──────────────────────────────────────────────────────────────

@router.get("/", response_model=list[ProjectOut])
async def list_projects(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[Project]:
    if current_user.role == Role.admin:
        q = select(Project).options(selectinload(Project.documents)).order_by(Project.id.desc())
    else:
        q = (
            select(Project)
            .options(selectinload(Project.documents))
            .where(Project.owner_id == current_user.id)
            .order_by(Project.id.desc())
        )
    result = await db.execute(q)
    projects = result.scalars().all()
    for p in projects:
        p.document_count = len(p.documents)
    return projects


@router.post("/", response_model=ProjectOut, status_code=status.HTTP_201_CREATED)
async def create_project(
    body: ProjectCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Project:
    project = Project(
        title=body.title,
        description=body.description,
        owner_id=current_user.id,
    )
    db.add(project)
    await db.flush()
    await log_event(db, current_user, AuditAction.project_create,
                    resource_type="project", resource_id=project.id, detail=body.title)
    project.document_count = 0
    return project


@router.get("/{project_id}", response_model=ProjectOut)
async def get_project(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Project:
    project = await _get_project_or_404(project_id, db, current_user)
    project.document_count = len(project.documents)
    return project


@router.put("/{project_id}", response_model=ProjectOut)
async def update_project(
    project_id: int,
    body: ProjectUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Project:
    project = await _get_project_or_404(project_id, db, current_user)
    if body.title is not None:
        project.title = body.title
    if body.description is not None:
        project.description = body.description
    if body.status is not None:
        project.status = body.status
    db.add(project)
    project.document_count = len(project.documents)
    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    project = await _get_project_or_404(project_id, db, current_user)
    await log_event(db, current_user, AuditAction.project_delete,
                    resource_type="project", resource_id=project_id, detail=project.title)
    # Remove files from MinIO
    for doc in project.documents:
        await delete_object(doc.storage_key)
    await db.delete(project)


# ── Document upload / list / delete ───────────────────────────────────────────

@router.post("/{project_id}/documents", response_model=DocumentOut,
             status_code=status.HTTP_201_CREATED)
async def upload_document(
    project_id: int,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Document:
    project = await _get_project_or_404(project_id, db, current_user)

    storage_key = f"projects/{project.id}/{file.filename}"
    content = await file.read()
    size = len(content)
    await upload_fileobj(storage_key, content, file.content_type or "application/octet-stream")

    doc = Document(
        filename=file.filename,
        content_type=file.content_type or "application/octet-stream",
        storage_key=storage_key,
        size_bytes=size,
        project_id=project.id,
    )
    db.add(doc)
    await db.flush()
    await log_event(db, current_user, AuditAction.document_upload,
                    resource_type="document", resource_id=doc.id, detail=file.filename)
    return doc


@router.get("/{project_id}/documents", response_model=list[DocumentOut])
async def list_documents(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[Document]:
    project = await _get_project_or_404(project_id, db, current_user)
    return project.documents


@router.delete("/{project_id}/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Documento non trovato.")
    await log_event(db, current_user, AuditAction.document_delete,
                    resource_type="document", resource_id=doc_id, detail=doc.filename)
    await delete_object(doc.storage_key)
    await db.delete(doc)
