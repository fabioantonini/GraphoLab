"""
GraphoLab Backend — Authentication router.

Endpoints:
  POST /auth/login     → access + refresh tokens (OAuth2 password flow)
  POST /auth/refresh   → new access token from refresh token
  POST /auth/logout    → client-side only (stateless JWT)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from jose import JWTError
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.audit import log_event
from backend.auth.jwt import TokenType, create_access_token, create_refresh_token, decode_token
from backend.auth.password import verify_password
from backend.database import get_db
from backend.models.audit import AuditAction
from backend.models.user import User

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
