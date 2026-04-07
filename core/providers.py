"""
GraphoLab — Provider abstraction layer.

Centralises detection of LLM/VLM/Embedding provider (Ollama vs OpenAI) and
exposes a ready-made OpenAI client.  All core modules import from here instead
of hard-coding Ollama assumptions.
"""

from __future__ import annotations

from pathlib import Path
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

_ENV_FILE = Path(__file__).parent.parent / ".env"

# ── Provider detection ────────────────────────────────────────────────────────

def is_openai_model(model_name: str) -> bool:
    """Return True if *model_name* refers to an OpenAI model."""
    return any(model_name.startswith(p) for p in OPENAI_MODEL_PREFIXES)


def embed_dim_for(model: str) -> int:
    """Return the embedding vector dimension for *model*."""
    return EMBED_DIMS.get(model, 768)

# ── API key management ────────────────────────────────────────────────────────

def _read_openai_key() -> str:
    """Return the OpenAI API key from env or .env file, empty string if absent."""
    import os
    val = os.environ.get("OPENAI_API_KEY", "").strip()
    if val:
        return val
    try:
        if _ENV_FILE.exists():
            for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    v = line.split("=", 1)[1].strip()
                    if v:
                        return v
    except Exception:
        pass
    return ""


def openai_key_configured() -> bool:
    """Return True if a non-empty OpenAI API key is available."""
    return bool(_read_openai_key())


def persist_openai_key(key: str) -> None:
    """Write (or update) OPENAI_API_KEY=<key> in the .env file."""
    _write_env_key("OPENAI_API_KEY", key)


def _write_env_key(env_key: str, value: str) -> None:
    """Update or append *env_key*=*value* in the .env file."""
    try:
        if _ENV_FILE.exists():
            lines = _ENV_FILE.read_text(encoding="utf-8").splitlines(keepends=True)
            new_lines: list[str] = []
            found = False
            for line in lines:
                if line.startswith(f"{env_key}="):
                    new_lines.append(f"{env_key}={value}\n")
                    found = True
                else:
                    new_lines.append(line)
            if not found:
                # ensure trailing newline before appending
                if new_lines and not new_lines[-1].endswith("\n"):
                    new_lines[-1] += "\n"
                new_lines.append(f"{env_key}={value}\n")
            _ENV_FILE.write_text("".join(new_lines), encoding="utf-8")
        else:
            _ENV_FILE.write_text(f"{env_key}={value}\n", encoding="utf-8")
    except Exception:
        pass


# ── OpenAI client ─────────────────────────────────────────────────────────────

_openai_client: Optional[object] = None


def get_openai_client():
    """
    Return a cached ``openai.OpenAI`` client instance.

    Raises ``RuntimeError`` if no API key is configured.
    The ``openai`` package is imported lazily so the app boots without it.
    """
    global _openai_client
    key = _read_openai_key()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY non configurata. "
            "Inserisci la chiave nella sezione Configurazione."
        )
    # Re-create the client only if the key changed (e.g. after persist_openai_key)
    if _openai_client is None or getattr(_openai_client, "api_key", None) != key:
        from openai import OpenAI  # lazy import
        _openai_client = OpenAI(api_key=key)
    return _openai_client


def invalidate_openai_client() -> None:
    """Force re-creation of the cached client (call after key change)."""
    global _openai_client
    _openai_client = None


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
