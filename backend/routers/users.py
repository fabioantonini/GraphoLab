"""
GraphoLab Backend — Users router.

Endpoints:
  GET  /users/me          → current user profile
  PUT  /users/me          → update own profile / password
  GET  /users/            → list all users (admin only)
  POST /users/            → create a new user (admin only)
  DELETE /users/{id}      → deactivate a user (admin only)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth.dependencies import get_current_user, require_role
from backend.auth.password import hash_password, verify_password
from backend.database import get_db
from backend.models.user import Role, User
from backend.models.user_settings import UserSettings

router = APIRouter(prefix="/users", tags=["users"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class UserOut(BaseModel):
    id: int
    email: str
    full_name: str
    role: Role
    is_active: bool
    organization_id: int | None

    model_config = {"from_attributes": True}


class UserCreate(BaseModel):
    email: EmailStr
    full_name: str
    password: str
    role: Role = Role.examiner
    organization_id: int | None = None


class UserUpdate(BaseModel):
    full_name: str | None = None
    current_password: str | None = None
    new_password: str | None = None


class UserSettingsOut(BaseModel):
    openai_key_configured: bool
    rag_model: str | None = None
    vlm_model: str | None = None
    ocr_model: str | None = None
    embed_model: str | None = None


class UserSettingsUpdate(BaseModel):
    openai_api_key: str


class UserModelsUpdate(BaseModel):
    rag_model: str | None = None
    vlm_model: str | None = None
    ocr_model: str | None = None
    embed_model: str | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/me", response_model=UserOut)
async def get_me(current_user: User = Depends(get_current_user)) -> User:
    return current_user


@router.put("/me", response_model=UserOut)
async def update_me(
    body: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> User:
    if body.full_name:
        current_user.full_name = body.full_name

    if body.new_password:
        if not body.current_password:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Fornisci la password attuale per cambiarla.",
            )
        if not verify_password(body.current_password, current_user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Password attuale non corretta.",
            )
        current_user.hashed_password = hash_password(body.new_password)

    db.add(current_user)
    return current_user


@router.get("/", response_model=list[UserOut], dependencies=[Depends(require_role(Role.admin))])
async def list_users(db: AsyncSession = Depends(get_db)) -> list[User]:
    result = await db.execute(select(User).order_by(User.id))
    return result.scalars().all()


@router.post("/", response_model=UserOut, status_code=status.HTTP_201_CREATED,
             dependencies=[Depends(require_role(Role.admin))])
async def create_user(
    body: UserCreate,
    db: AsyncSession = Depends(get_db),
) -> User:
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Email '{body.email}' già registrata.",
        )
    user = User(
        email=body.email,
        full_name=body.full_name,
        hashed_password=hash_password(body.password),
        role=body.role,
        organization_id=body.organization_id,
    )
    db.add(user)
    await db.flush()
    return user


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT,
               dependencies=[Depends(require_role(Role.admin))])
async def deactivate_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Non puoi disattivare il tuo stesso account.",
        )
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Utente non trovato.")
    user.is_active = False
    db.add(user)


# ── Per-user settings endpoints ───────────────────────────────────────────────

async def _get_or_create_settings(user_id: int, db: AsyncSession) -> UserSettings:
    """Return the UserSettings row for *user_id*, creating it if absent."""
    result = await db.execute(select(UserSettings).where(UserSettings.user_id == user_id))
    row = result.scalar_one_or_none()
    if row is None:
        row = UserSettings(user_id=user_id)
        db.add(row)
        await db.flush()
    return row


@router.get("/me/settings", response_model=UserSettingsOut)
async def get_my_settings(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> UserSettingsOut:
    """Return the current user's personal settings: OpenAI key status and saved model preferences."""
    result = await db.execute(
        select(UserSettings).where(UserSettings.user_id == current_user.id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        return UserSettingsOut(openai_key_configured=False)
    return UserSettingsOut(
        openai_key_configured=bool(row.openai_api_key_enc),
        rag_model=row.rag_model,
        vlm_model=row.vlm_model,
        ocr_model=row.ocr_model,
        embed_model=row.embed_model,
    )


@router.put("/me/settings", response_model=UserSettingsOut)
async def save_my_openai_key(
    body: UserSettingsUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> UserSettingsOut:
    """Validate and save the user's personal OpenAI API key (encrypted in DB)."""
    from core.providers import validate_openai_key, encrypt_key
    try:
        valid = validate_openai_key(body.openai_api_key)
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chiave OpenAI non valida o non autorizzata.",
        )
    row = await _get_or_create_settings(current_user.id, db)
    row.openai_api_key_enc = encrypt_key(body.openai_api_key)
    db.add(row)
    return UserSettingsOut(openai_key_configured=True)


@router.delete("/me/settings/openai-key", status_code=status.HTTP_204_NO_CONTENT)
async def delete_my_openai_key(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    """Remove the user's personal OpenAI API key."""
    result = await db.execute(
        select(UserSettings).where(UserSettings.user_id == current_user.id)
    )
    row = result.scalar_one_or_none()
    if row is not None:
        row.openai_api_key_enc = None
        db.add(row)


@router.patch("/me/settings/models", response_model=UserSettingsOut)
async def save_my_model_preferences(
    body: UserModelsUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> UserSettingsOut:
    """Persist the user's model selections so they are restored on next login."""
    row = await _get_or_create_settings(current_user.id, db)
    if body.rag_model is not None:
        row.rag_model = body.rag_model
    if body.vlm_model is not None:
        row.vlm_model = body.vlm_model
    if body.ocr_model is not None:
        row.ocr_model = body.ocr_model
    if body.embed_model is not None:
        row.embed_model = body.embed_model
    db.add(row)
    return UserSettingsOut(
        openai_key_configured=bool(row.openai_api_key_enc),
        rag_model=row.rag_model,
        vlm_model=row.vlm_model,
        ocr_model=row.ocr_model,
        embed_model=row.embed_model,
    )
