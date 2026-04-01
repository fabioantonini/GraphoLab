"""
GraphoLab Backend — Audit log helper.

Usage:
    from backend.audit import log_event
    from backend.models.audit import AuditAction

    await log_event(
        db       = db,
        user     = current_user,
        action   = AuditAction.analysis_run,
        resource_type = "analysis",
        resource_id   = analysis.id,
        detail        = "pipeline",
        ip_address    = request.client.host if request.client else None,
    )

The function simply inserts a row and flushes — the caller's db session
commits at the end of the request as usual.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.audit import AuditAction, AuditLog
from backend.models.user import User


async def log_event(
    db: AsyncSession,
    user: User,
    action: AuditAction,
    resource_type: str | None = None,
    resource_id: int | None = None,
    detail: str | None = None,
    ip_address: str | None = None,
) -> None:
    """Append one row to the audit_log table.  Never raises — failures are silently swallowed
    so that a logging error never breaks an otherwise successful operation."""
    try:
        entry = AuditLog(
            user_id=user.id,
            user_email=user.email,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            detail=detail,
            ip_address=ip_address,
        )
        db.add(entry)
        await db.flush()
    except Exception:
        pass  # audit must never break the main flow
