"""
GraphoLab Backend — AuditLog ORM model.

Append-only table: no UPDATE or DELETE is ever issued by application code.
Every sensitive operation is recorded here for forensic chain-of-custody.
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class AuditAction(str, enum.Enum):
    login                = "login"
    project_create       = "project_create"
    project_delete       = "project_delete"
    document_upload      = "document_upload"
    document_delete      = "document_delete"
    analysis_run         = "analysis_run"
    analysis_clear       = "analysis_clear"
    pdf_download         = "pdf_download"


class AuditLog(Base):
    """Immutable audit trail.  Never UPDATE or DELETE rows from this table."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # UTC timestamp — server-generated, not client-supplied
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    # Who performed the action (snapshot: user may be deleted later)
    user_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    user_email: Mapped[str] = mapped_column(String(256), nullable=False)

    # What happened
    action: Mapped[AuditAction] = mapped_column(Enum(AuditAction), nullable=False, index=True)

    # On which resource (e.g. resource_type="project", resource_id=42)
    resource_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resource_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Optional extra context (e.g. analysis type, filename)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Network info
    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
