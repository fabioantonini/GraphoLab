"""
Shared FastAPI dependencies for per-request OpenAI key propagation.
"""

from __future__ import annotations

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth.dependencies import get_current_user
from backend.database import get_db
from backend.models.user import User
from backend.models.user_settings import UserSettings


async def set_openai_context(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Resolve the current user's OpenAI key and inject it into the
    per-request ContextVar so all core module calls use the correct key.

    Key priority:
      1. User's personal key (decrypted from UserSettings)
      2. Global OPENAI_API_KEY env var (fallback, handled inside get_openai_client)

    Add to any endpoint that may call OpenAI::

        @router.post("/something")
        async def endpoint(
            _: None = Depends(set_openai_context),
            current_user: User = Depends(get_current_user),
        ):
            ...
    """
    from core.providers import decrypt_key, set_request_api_key

    result = await db.execute(
        select(UserSettings).where(UserSettings.user_id == current_user.id)
    )
    row = result.scalar_one_or_none()
    key: str | None = None
    if row is not None and row.openai_api_key_enc:
        key = decrypt_key(row.openai_api_key_enc)
    set_request_api_key(key or None)
