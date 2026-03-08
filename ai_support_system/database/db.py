"""Подключение к базе данных."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .models import Base

logger = logging.getLogger(__name__)


def get_engine(database_url: str):
    """Создать асинхронный движок SQLAlchemy."""
    return create_async_engine(
        database_url,
        echo=False,
        future=True,
    )


def get_session_factory(engine):
    """Создать фабрику сессий."""
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )


async def init_db(database_url: str, data_dir: Path = None) -> async_sessionmaker:
    """Инициализация базы данных."""
    if data_dir:
        data_dir.mkdir(parents=True, exist_ok=True)
    engine = get_engine(database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = get_session_factory(engine)
    logger.info("Database initialized")
    return session_factory


@asynccontextmanager
async def get_session(session_factory: async_sessionmaker):
    """Контекстный менеджер для получения сессии."""
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
