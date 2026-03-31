"""
GraphoLab Backend — MinIO async storage client.

Wraps the synchronous `minio` SDK in asyncio-friendly helpers using
`anyio.to_thread.run_sync` to avoid blocking the event loop.

Public API:
  upload_fileobj(key, data, content_type)  → None
  download_object(key)                     → bytes
  delete_object(key)                       → None
  get_presigned_url(key, expires_seconds)  → str
"""

from __future__ import annotations

import io
from datetime import timedelta

import anyio
from minio import Minio
from minio.error import S3Error

from backend.config import settings

# ── Singleton client ──────────────────────────────────────────────────────────

_client: Minio | None = None


def _get_client() -> Minio:
    global _client
    if _client is None:
        _client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        # Ensure the bucket exists
        if not _client.bucket_exists(settings.minio_bucket):
            _client.make_bucket(settings.minio_bucket)
    return _client


# ── Sync helpers (run in thread pool) ────────────────────────────────────────

def _upload_sync(key: str, data: bytes, content_type: str) -> None:
    client = _get_client()
    client.put_object(
        settings.minio_bucket,
        key,
        io.BytesIO(data),
        length=len(data),
        content_type=content_type,
    )


def _download_sync(key: str) -> bytes:
    client = _get_client()
    response = client.get_object(settings.minio_bucket, key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def _delete_sync(key: str) -> None:
    client = _get_client()
    try:
        client.remove_object(settings.minio_bucket, key)
    except S3Error:
        pass  # already deleted or never existed — no-op


def _presigned_sync(key: str, expires_seconds: int) -> str:
    client = _get_client()
    return client.presigned_get_object(
        settings.minio_bucket,
        key,
        expires=timedelta(seconds=expires_seconds),
    )


# ── Async public API ──────────────────────────────────────────────────────────

async def upload_fileobj(key: str, data: bytes, content_type: str) -> None:
    await anyio.to_thread.run_sync(lambda: _upload_sync(key, data, content_type))


async def download_object(key: str) -> bytes:
    return await anyio.to_thread.run_sync(lambda: _download_sync(key))


async def delete_object(key: str) -> None:
    await anyio.to_thread.run_sync(lambda: _delete_sync(key))


async def get_presigned_url(key: str, expires_seconds: int = 3600) -> str:
    return await anyio.to_thread.run_sync(lambda: _presigned_sync(key, expires_seconds))
