"""
Главная точка входа FAQ Telegram Assistant.
Запуск: python main.py
"""
import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from core.config import settings
from database.db import init_db
from database.models import FAQEntry
from services.clustering_service import ClusteringService
from services.embedding_service import EmbeddingService
from services.faq_service import FAQService
from services.learning_service import LearningService
from services.pending_service import PendingService
from vectorstore.faiss_index import FAISSIndex

from api.routes import router as api_router
from admin.admin_routes import router as admin_router

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


async def rebuild_faiss_index(session_factory, embedding_service, faiss_index):
    """Пересборка FAISS индекса из базы данных."""
    from sqlalchemy import select

    async with session_factory() as session:
        result = await session.execute(select(FAQEntry))
        entries = list(result.scalars().all())

    if not entries:
        logger.info("No FAQ entries for rebuild")
        return

    questions = [e.question for e in entries]
    ids = [e.id for e in entries]
    embeddings = embedding_service.embed_batch(questions)
    faiss_index.rebuild(embeddings, ids)
    faiss_index.save()
    logger.info("FAISS index rebuilt with %d entries", len(entries))


async def _background_init(app: FastAPI):
    """Фоновая загрузка модели и пересборка индекса."""
    try:
        embedding_service = app.state.embedding_service
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, embedding_service._ensure_loaded)
        faiss_index = app.state.faiss_index
        session_factory = app.state.session_factory
        if faiss_index.ntotal == 0:
            await rebuild_faiss_index(session_factory, embedding_service, faiss_index)
    except Exception as e:
        logger.exception("Background init failed: %s", e)
    finally:
        app.state.ready = True
        logger.info("Application fully ready")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Жизненный цикл приложения."""
    logger.info("Starting application...")
    app.state.ready = False

    # Директория для данных
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Инициализация БД
    session_factory = await init_db(
        settings.DATABASE_URL,
        settings.DATA_DIR,
    )
    app.state.session_factory = session_factory

    # Эмбеддинги (ленивая загрузка — сервер стартует сразу)
    embedding_service = EmbeddingService(settings.EMBEDDING_MODEL)
    app.state.embedding_service = embedding_service

    # FAISS индекс (размерность без загрузки модели)
    faiss_path = Path(settings.FAISS_INDEX_PATH)
    faiss_index = FAISSIndex(
        dimension=embedding_service.dimension,
        index_path=faiss_path,
    )
    faiss_index.load()
    app.state.faiss_index = faiss_index

    # Сервисы
    app.state.faq_service = FAQService(
        embedding_service=embedding_service,
        faiss_index=faiss_index,
        session_factory=session_factory,
        similarity_threshold=settings.SIMILARITY_THRESHOLD,
        cache_ttl=settings.CACHE_TTL_SECONDS,
        cache_max_size=settings.CACHE_MAX_SIZE,
    )
    app.state.pending_service = PendingService(session_factory)
    app.state.learning_service = LearningService(
        embedding_service=embedding_service,
        faiss_index=faiss_index,
        session_factory=session_factory,
    )
    app.state.clustering_service = ClusteringService(
        eps=settings.CLUSTERING_EPS,
        min_samples=settings.CLUSTERING_MIN_SAMPLES,
    )

    asyncio.create_task(_background_init(app))
    logger.info("Application started (model loading in background)")
    yield

    # Shutdown
    logger.info("Shutting down...")


def create_app() -> FastAPI:
    """Создание FastAPI приложения."""
    app = FastAPI(
        title="FAQ Telegram Assistant API",
        description="Self-learning FAQ system with Telegram bot",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.include_router(api_router)
    app.include_router(admin_router)

    # Статика если есть
    static_path = Path(__file__).parent / "static"
    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

    return app


app = create_app()


async def run_bot_task():
    """Запуск Telegram бота в фоне."""
    if not settings.TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN not set, bot disabled")
        return

    logger.info("Telegram bot starting...")
    api_url = f"http://127.0.0.1:{settings.API_PORT}"
    from bot.telegram_bot import run_bot

    await run_bot(settings.TELEGRAM_BOT_TOKEN, api_url)


if __name__ == "__main__":
    import uvicorn

    async def run_all():
        """Запуск API и бота параллельно."""
        config = uvicorn.Config(
            app,
            host=settings.API_HOST,
            port=settings.API_PORT,
            log_level="info",
        )
        server = uvicorn.Server(config)
        api_task = asyncio.create_task(server.serve())
        bot_task = asyncio.create_task(run_bot_task())
        await asyncio.gather(api_task, bot_task)

    asyncio.run(run_all())
