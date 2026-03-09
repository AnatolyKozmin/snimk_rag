"""
Главная точка входа FAQ Telegram Assistant.
Запуск: python main.py
"""
import asyncio
import logging
import os
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
from services.llm_service import LLMService
from services.pending_service import PendingService
from vectorstore.faiss_index import FAISSIndex

from api.routes import router as api_router
from admin.admin_routes import router as admin_router

# Настройка логирования (должно быть до импорта app)
from core.logging_config import setup_logging

setup_logging(service_name="api")
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


async def _import_initial_faq(app: FastAPI) -> int:
    """Автоимпорт FAQ из Excel при старте. Возвращает количество импортированных записей."""
    from services.faq_loader import load_faq_from_bytes

    project_root = Path(__file__).parent
    # Порядок: data, faq_import (volumes в Docker), корень проекта
    candidates = [
        settings.DATA_DIR / "faq.xlsx",
        settings.DATA_DIR / "faq.csv",
        project_root / "faq_import" / "faq.xlsx",
        project_root / "faq_import" / "faq.csv",
        project_root / settings.INITIAL_FAQ_FILE,
        project_root / "faq.xlsx",
        project_root / "faq.csv",
    ]
    faq_path = None
    for p in candidates:
        if p.exists():
            faq_path = p
            break

    if not faq_path:
        logger.info("Автоимпорт: файл faq.xlsx/faq.csv не найден, пропуск")
        return 0

    logger.info("Автоимпорт: найден файл %s", faq_path)
    try:
        content = faq_path.read_bytes()
        pairs = load_faq_from_bytes(content, faq_path.name)
    except Exception as e:
        logger.error("Автоимпорт: ошибка чтения файла: %s", e)
        return 0

    if not pairs:
        logger.warning("Автоимпорт: файл пуст или неверный формат")
        return 0

    logger.info("Автоимпорт: загружено %d пар Q/A, начинаю добавление в FAQ", len(pairs))
    learning_service = app.state.learning_service
    faq_service = app.state.faq_service

    for i, (question, answer) in enumerate(pairs, 1):
        try:
            await learning_service.add_qa(question, answer)
            if i % 10 == 0 or i == len(pairs):
                logger.info("Автоимпорт: добавлено %d/%d записей", i, len(pairs))
        except Exception as e:
            logger.error("Автоимпорт: ошибка в строке %d (%s): %s", i, question[:50], e)

    faq_service.invalidate_cache()
    logger.info("Автоимпорт: завершён, импортировано %d записей", len(pairs))
    return len(pairs)


async def _background_init(app: FastAPI):
    """Фоновая загрузка модели, пересборка индекса и автоимпорт FAQ."""
    try:
        logger.info("Фоновая инициализация: загрузка модели эмбеддингов...")
        embedding_service = app.state.embedding_service
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, embedding_service._ensure_loaded)
        logger.info("Фоновая инициализация: модель эмбеддингов загружена")

        if settings.USE_LLM_RAG and app.state.llm_service:
            logger.info("Фоновая инициализация: загрузка LLM...")
            await loop.run_in_executor(None, app.state.llm_service._ensure_loaded)
            logger.info("Фоновая инициализация: LLM загружен")

        faiss_index = app.state.faiss_index
        session_factory = app.state.session_factory

        if faiss_index.ntotal == 0:
            logger.info("Фоновая инициализация: FAISS индекс пуст, пересборка из БД...")
            await rebuild_faiss_index(session_factory, embedding_service, faiss_index)
        else:
            logger.info("Фоновая инициализация: FAISS индекс загружен (%d векторов)", faiss_index.ntotal)

        # Автоимпорт FAQ из Excel
        imported = await _import_initial_faq(app)
        if imported > 0:
            logger.info("Фоновая инициализация: после импорта пересобираю FAISS индекс")
            await rebuild_faiss_index(session_factory, embedding_service, faiss_index)

    except Exception as e:
        logger.exception("Фоновая инициализация: ошибка: %s", e)
    finally:
        app.state.ready = True
        logger.info("Фоновая инициализация: завершена, сервис готов")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Жизненный цикл приложения."""
    logger.info("Starting application...")
    app.state.ready = False

    # Кэш моделей (для локального запуска; в Docker HF_HOME задаётся в env)
    if "HF_HOME" not in os.environ:
        cache_dir = str(settings.MODEL_CACHE_DIR.absolute())
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir

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

    # LLM (опционально)
    llm_service = None
    if settings.USE_LLM_RAG:
        cache_dir = os.environ.get("HF_HOME")
        llm_service = LLMService(
            model_name=settings.LLM_MODEL,
            cache_dir=cache_dir,
        )
    app.state.llm_service = llm_service

    # Сервисы
    app.state.faq_service = FAQService(
        embedding_service=embedding_service,
        faiss_index=faiss_index,
        session_factory=session_factory,
        similarity_threshold=settings.SIMILARITY_THRESHOLD,
        similarity_threshold_llm=settings.SIMILARITY_THRESHOLD_LLM,
        cache_ttl=settings.CACHE_TTL_SECONDS,
        cache_max_size=settings.CACHE_MAX_SIZE,
        llm_service=llm_service,
        use_llm_rag=settings.USE_LLM_RAG,
        llm_top_k=settings.LLM_TOP_K,
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

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level="info",
    )
