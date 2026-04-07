"""
GraphoLab Backend — UserSettings ORM model.

Stores per-user configuration, starting with the OpenAI API key
encrypted with Fernet symmetric encryption.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


class UserSettings(Base):
    __tablename__ = "user_settings"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )
    # Fernet-encrypted OpenAI API key (base64 ciphertext).
    # None means the user has not configured a personal key.
    openai_api_key_enc: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Last-selected AI models (None = use server default)
    rag_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    vlm_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    ocr_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    embed_model: Mapped[str | None] = mapped_column(String(128), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user: Mapped["User"] = relationship("User", back_populates="settings")  # type: ignore[name-defined]
