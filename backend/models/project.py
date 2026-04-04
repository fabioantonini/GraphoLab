"""
GraphoLab Backend — Project and Analysis ORM models.

A Project (= "Perizia") groups one or more Documents and their Analyses.
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


class ProjectStatus(str, enum.Enum):
    draft = "draft"
    in_progress = "in_progress"
    completed = "completed"
    archived = "archived"


class AnalysisType(str, enum.Enum):
    htr = "htr"
    signature_detection = "signature_detection"
    signature_verification = "signature_verification"
    ner = "ner"
    writer_identification = "writer_identification"
    graphology = "graphology"
    pipeline = "pipeline"
    dating = "dating"
    rag = "rag"


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[ProjectStatus] = mapped_column(
        Enum(ProjectStatus), default=ProjectStatus.draft, nullable=False
    )

    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    owner: Mapped["User"] = relationship("User", back_populates="projects")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    documents: Mapped[list[Document]] = relationship(
        "Document", back_populates="project", cascade="all, delete-orphan"
    )
    analyses: Mapped[list[Analysis]] = relationship(
        "Analysis", back_populates="project", cascade="all, delete-orphan"
    )
    agent_chats: Mapped[list[AgentChat]] = relationship(
        "AgentChat", back_populates="project", cascade="all, delete-orphan"
    )


class Document(Base):
    """A file uploaded to a project (stored in MinIO)."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str] = mapped_column(String(128), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False)  # MinIO object key
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)

    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    project: Mapped[Project] = relationship("Project", back_populates="documents")

    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Analysis(Base):
    """Result of one AI analysis step on a project document."""

    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    analysis_type: Mapped[AnalysisType] = mapped_column(Enum(AnalysisType), nullable=False)
    result_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_storage_key: Mapped[str | None] = mapped_column(
        String(512), nullable=True  # MinIO key for image/PDF result
    )

    document_id: Mapped[int | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"), nullable=True
    )
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    project: Mapped[Project] = relationship("Project", back_populates="analyses")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class AgentChat(Base):
    """A persistent chat session inside an agent project."""

    __tablename__ = "agent_chats"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(256), nullable=False, default="Nuova chat")

    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    project: Mapped[Project] = relationship("Project", back_populates="agent_chats")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    messages: Mapped[list[AgentMessage]] = relationship(
        "AgentMessage", back_populates="chat", cascade="all, delete-orphan",
        order_by="AgentMessage.created_at"
    )


class AgentMessage(Base):
    """A single message in an AgentChat."""

    __tablename__ = "agent_messages"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    chat_id: Mapped[int] = mapped_column(
        ForeignKey("agent_chats.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chat: Mapped[AgentChat] = relationship("AgentChat", back_populates="messages")

    role: Mapped[str] = mapped_column(String(16), nullable=False)   # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # JSON-encoded list of Document.id attached by the user (user messages only)
    file_ids: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
