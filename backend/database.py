"""
GraphoLab Backend — Async SQLAlchemy database setup.
"""

from __future__ import annotations

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from backend.config import settings

# SQLite needs check_same_thread=False; pool_pre_ping is not compatible with aiosqlite
_is_sqlite = settings.database_url.startswith("sqlite")
_engine_kwargs: dict = {"echo": settings.debug}
if _is_sqlite:
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    _engine_kwargs["pool_pre_ping"] = True

engine = create_async_engine(settings.database_url, **_engine_kwargs)

if _is_sqlite:
    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragmas(dbapi_conn, _conn_record):
        """Enable WAL mode and busy timeout on every new SQLite connection.

        WAL mode: allows concurrent reads + 1 writer without locking errors.
        busy_timeout: SQLite waits up to 10s before raising 'database is locked'.
        """
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=10000")
        cursor.close()

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


async def get_db() -> AsyncSession:
    """FastAPI dependency: yields an async DB session per request."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create all tables and apply incremental column migrations."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _migrate(conn)


async def _migrate(conn) -> None:
    """Add new columns to existing tables without touching existing data.

    SQLite does not support IF NOT EXISTS in ALTER TABLE, so we catch the
    OperationalError that fires when a column already exists and continue.
    PostgreSQL supports ADD COLUMN IF NOT EXISTS natively.
    """
    is_sqlite = settings.database_url.startswith("sqlite")

    new_columns = [
        ("user_settings", "rag_model",   "VARCHAR(128)"),
        ("user_settings", "vlm_model",   "VARCHAR(128)"),
        ("user_settings", "ocr_model",   "VARCHAR(64)"),
        ("user_settings", "embed_model", "VARCHAR(128)"),
    ]

    for table, column, col_type in new_columns:
        if is_sqlite:
            try:
                await conn.execute(
                    text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                )
            except Exception:
                pass  # column already exists — safe to ignore
        else:
            await conn.execute(
                text(
                    f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {col_type}"
                )
            )
