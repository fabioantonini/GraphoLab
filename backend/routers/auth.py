"""
GraphoLab Backend — Authentication router.

Endpoints:
  POST /auth/login                    → access + refresh tokens (OAuth2 password flow)
  POST /auth/refresh                  → new access token from refresh token
  POST /auth/logout                   → client-side only (stateless JWT)
  POST /auth/reset-password/generate  → admin generates a reset token for a user
  POST /auth/reset-password/confirm   → user sets a new password via reset token
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from jose import JWTError
from pydantic import BaseModel, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.audit import log_event
from backend.auth.dependencies import get_current_user
from backend.auth.jwt import TokenType, create_access_token, create_refresh_token, create_reset_token, decode_token
from backend.auth.password import hash_password, verify_password
from backend.database import get_db
from backend.models.audit import AuditAction
from backend.models.user import Role, User

router = APIRouter(prefix="/auth", tags=["auth"])


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/login", response_model=TokenResponse)
async def login(
    request: Request,
    form: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    result = await db.execute(select(User).where(User.email == form.username))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(form.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email o password non corretti.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account disabilitato. Contatta l'amministratore.",
        )

    await log_event(
        db, user, AuditAction.login,
        ip_address=request.client.host if request.client else None,
    )

    return TokenResponse(
        access_token=create_access_token(user.id, user.role),
        refresh_token=create_refresh_token(user.id),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    body: RefreshRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Refresh token non valido o scaduto.",
    )
    try:
        payload = decode_token(body.refresh_token)
        if payload.get("type") != TokenType.refresh:
            raise credentials_exc
        user_id = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        raise credentials_exc

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None or not user.is_active:
        raise credentials_exc

    return TokenResponse(
        access_token=create_access_token(user.id, user.role),
        refresh_token=create_refresh_token(user.id),
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout() -> None:
    """JWT is stateless — logout is handled client-side by discarding the token."""
    pass


# ── Password reset (admin-mediated, no SMTP required) ─────────────────────────

class ResetGenerateRequest(BaseModel):
    user_id: int


class ResetGenerateResponse(BaseModel):
    token: str


class ResetConfirmRequest(BaseModel):
    token: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("La password deve essere di almeno 8 caratteri.")
        return v


@router.post(
    "/reset-password/generate",
    response_model=ResetGenerateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def generate_reset_token(
    body: ResetGenerateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ResetGenerateResponse:
    """Admin-only: generate a 24h reset token for the given user."""
    if current_user.role != Role.admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Accesso riservato agli amministratori.")

    result = await db.execute(select(User).where(User.id == body.user_id))
    target = result.scalar_one_or_none()
    if target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Utente non trovato.")

    token = create_reset_token(target.id)

    await log_event(
        db, current_user, AuditAction.password_reset_generate,
        resource_type="user", resource_id=target.id,
        detail=f"Reset token generato per {target.email}",
        ip_address=request.client.host if request.client else None,
    )

    return ResetGenerateResponse(token=token)


@router.post("/reset-password/confirm", status_code=status.HTTP_204_NO_CONTENT)
async def confirm_reset(
    body: ResetConfirmRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Public: consume a reset token and set a new password."""
    invalid_exc = HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Token non valido o scaduto.",
    )
    try:
        payload = decode_token(body.token)
        if payload.get("type") != TokenType.reset:
            raise invalid_exc
        user_id = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        raise invalid_exc

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None or not user.is_active:
        raise invalid_exc

    user.hashed_password = hash_password(body.new_password)
    await db.flush()

    await log_event(
        db, user, AuditAction.password_reset_confirm,
        resource_type="user", resource_id=user.id,
        detail="Password reimpostata tramite token di reset",
        ip_address=request.client.host if request.client else None,
    )
