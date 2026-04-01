"""
GraphoLab Backend — Audit log router.

Endpoints:
  GET /audit   → paginated audit log (admin only)
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth.dependencies import get_current_user
from backend.database import get_db
from backend.models.audit import AuditAction, AuditLog
from backend.models.user import Role, User

router = APIRouter(prefix="/audit", tags=["audit"])


class AuditLogOut(BaseModel):
    id: int
    timestamp: datetime
    user_id: int | None
    user_email: str
    action: AuditAction
    resource_type: str | None
    resource_id: int | None
    detail: str | None
    ip_address: str | None

    model_config = {"from_attributes": True}


class AuditPage(BaseModel):
    total: int
    items: list[AuditLogOut]


@router.get("/", response_model=AuditPage)
async def list_audit_log(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    action: AuditAction | None = Query(None),
    user_email: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AuditPage:
    if current_user.role != Role.admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Solo gli amministratori possono visualizzare il log.")

    q = select(AuditLog)
    if action:
        q = q.where(AuditLog.action == action)
    if user_email:
        q = q.where(AuditLog.user_email.ilike(f"%{user_email}%"))

    count_q = select(func.count()).select_from(q.subquery())
    total = (await db.execute(count_q)).scalar_one()

    q = q.order_by(AuditLog.timestamp.desc()).offset((page - 1) * page_size).limit(page_size)
    rows = (await db.execute(q)).scalars().all()

    return AuditPage(total=total, items=list(rows))
