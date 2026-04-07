"""
GraphoLab — Provider abstraction layer.

Centralises detection of LLM/VLM/Embedding provider (Ollama vs OpenAI) and
exposes a ready-made OpenAI client.  All core modules import from here instead
of hard-coding Ollama assumptions.

Per-request API key propagation
────────────────────────────────
FastAPI routers set `set_request_api_key(key)` at the start of each request
via a dependency.  `get_openai_client()` picks it up automatically through a
`contextvars.ContextVar`, so core modules need no signature changes.

Priority order:
  1. Per-request key (from user's UserSettings row, decrypted)
  2. Global key from `OPENAI_API_KEY` env var (Docker / host env — never written to file)
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Optional

# ── Model catalogues ──────────────────────────────────────────────────────────

# Model ID prefixes that unambiguously identify an OpenAI model
OPENAI_MODEL_PREFIXES: tuple[str, ...] = ("gpt-", "text-embedding-3-", "o1-", "o3-")

# Latest models as of April 2026
OPENAI_LLM_MODELS: list[str] = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"]
OPENAI_VLM_MODELS: list[str] = ["gpt-5.4", "gpt-5.4-mini"]   # all gpt-5.4 have native vision
OPENAI_EMBED_MODELS: list[str] = ["text-embedding-3-small", "text-embedding-3-large"]

# Embedding output dimensions
EMBED_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "nomic-embed-text":        768,
}

# ── Provider detection ────────────────────────────────────────────────────────

def is_openai_model(model_name: str) -> bool:
    """Return True if *model_name* refers to an OpenAI model."""
    return any(model_name.startswith(p) for p in OPENAI_MODEL_PREFIXES)


def embed_dim_for(model: str) -> int:
    """Return the embedding vector dimension for *model*."""
    return EMBED_DIMS.get(model, 768)


# ── Per-request API key (ContextVar) ─────────────────────────────────────────

_request_api_key: ContextVar[str | None] = ContextVar("_request_api_key", default=None)


def set_request_api_key(key: str | None) -> None:
    """Set the OpenAI API key for the current async request context."""
    _request_api_key.set(key)


# ── API key management ────────────────────────────────────────────────────────

def _read_openai_key() -> str:
    """Return the global OpenAI API key from the environment only.

    The global key is set via the OPENAI_API_KEY environment variable
    (Docker, host shell, or CI). It is NEVER written to or read from a file —
    per-user keys are stored encrypted in the database.
    """
    return os.environ.get("OPENAI_API_KEY", "").strip()


def openai_key_configured() -> bool:
    """Return True if a global OpenAI API key is available in the environment."""
    return bool(_read_openai_key())


# ── Encryption helpers (Fernet) ───────────────────────────────────────────────

def _get_fernet():
    """Return a Fernet instance using SETTINGS_ENCRYPTION_KEY, or None if not configured."""
    try:
        from backend.config import settings
        key = settings.settings_encryption_key.strip()
    except Exception:
        key = os.environ.get("SETTINGS_ENCRYPTION_KEY", "").strip()
    if not key:
        return None
    try:
        from cryptography.fernet import Fernet
        return Fernet(key.encode() if isinstance(key, str) else key)
    except Exception:
        return None


def encrypt_key(plain: str) -> str:
    """Encrypt *plain* with Fernet.  Returns plain-text if no encryption key is configured."""
    f = _get_fernet()
    if f is None:
        return plain
    return f.encrypt(plain.encode()).decode()


def decrypt_key(enc: str) -> str:
    """Decrypt *enc* with Fernet.  Returns *enc* as-is if no encryption key is configured."""
    f = _get_fernet()
    if f is None:
        return enc
    try:
        return f.decrypt(enc.encode()).decode()
    except Exception:
        # Fallback: may be a plain-text key stored before encryption was enabled
        return enc


# ── OpenAI client ─────────────────────────────────────────────────────────────

_openai_client: Optional[object] = None
_openai_client_key: str = ""  # key used to build the cached client


def get_openai_client():
    """
    Return an ``openai.OpenAI`` client.

    Key resolution order:
      1. Per-request ContextVar (user's decrypted key, set by FastAPI dependency)
      2. Global ``OPENAI_API_KEY`` env var

    A new client is created whenever the resolved key changes so stale
    credentials are never reused.

    Raises ``RuntimeError`` if no API key is available.
    """
    global _openai_client, _openai_client_key

    key = _request_api_key.get() or _read_openai_key()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY non configurata. "
            "Inserisci la chiave nella sezione Configurazione."
        )

    # If the per-request ContextVar is set we always create a fresh client
    # (different users have different keys; caching across requests is unsafe).
    if _request_api_key.get():
        from openai import OpenAI
        return OpenAI(api_key=key)

    # Global key: cache to avoid re-creating on every call
    if _openai_client is None or _openai_client_key != key:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=key)
        _openai_client_key = key
    return _openai_client


def invalidate_openai_client() -> None:
    """Force re-creation of the cached global client (call after env key change)."""
    global _openai_client, _openai_client_key
    _openai_client = None
    _openai_client_key = ""


def validate_openai_key(key: str) -> bool:
    """
    Return True if *key* is a valid OpenAI API key.

    Makes a lightweight ``models.list()`` call to verify.
    Raises ``RuntimeError`` if the openai package is not installed.
    """
    try:
        from openai import OpenAI, AuthenticationError, PermissionDeniedError
    except ImportError as e:
        raise RuntimeError(
            "Il pacchetto 'openai' non è installato. "
            "Esegui: pip install 'openai>=2.0.0'"
        ) from e
    try:
        client = OpenAI(api_key=key, timeout=10.0)
        client.models.list()
        return True
    except (AuthenticationError, PermissionDeniedError):
        return False
    except Exception:
        # Network errors, timeouts, etc. — don't block the user
        return True
