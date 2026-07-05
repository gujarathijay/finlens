"""
SQLite database setup with SQLAlchemy async.

Stores every extraction request and result for auditing.
Swap to Postgres in production by changing DATABASE_URL — code stays the same.
"""

import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# ── Database engine ──

DATABASE_URL = "sqlite+aiosqlite:///finlens.db"

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


# ── Models ──


class Base(DeclarativeBase):
    pass


class Extraction(Base):
    """One extraction request + result."""

    __tablename__ = "extractions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())

    # Input
    input_text: Mapped[str] = mapped_column(Text)
    input_length: Mapped[int] = mapped_column(Integer)

    # Output
    output_json: Mapped[str] = mapped_column(Text)  # raw JSON string
    company_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    num_risks: Mapped[int] = mapped_column(Integer, default=0)
    num_events: Mapped[int] = mapped_column(Integer, default=0)
    num_obligations: Mapped[int] = mapped_column(Integer, default=0)

    # Quality
    guardrails_passed: Mapped[bool] = mapped_column(default=True)
    guardrail_failures: Mapped[str | None] = mapped_column(String(500), nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)

    # Status
    status: Mapped[str] = mapped_column(String(20), default="success")  # success, failed, flagged


async def init_db():
    """Create tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
