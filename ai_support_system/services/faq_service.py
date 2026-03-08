"""Сервис FAQ - поиск ответов по вопросам."""
import logging
from typing import List, Optional, Tuple

from cachetools import TTLCache
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import FAQEntry
from services.embedding_service import EmbeddingService
from services.normalizer import normalize_question
from vectorstore.faiss_index import FAISSIndex

logger = logging.getLogger(__name__)


class FAQService:
    """Сервис для работы с FAQ."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        faiss_index: FAISSIndex,
        session_factory,
        similarity_threshold: float = 0.8,
        cache_ttl: int = 300,
        cache_max_size: int = 1000,
    ):
        self.embedding_service = embedding_service
        self.faiss_index = faiss_index
        self.session_factory = session_factory
        self.similarity_threshold = similarity_threshold
        self._cache = TTLCache(maxsize=cache_max_size, ttl=cache_ttl)

    async def search(
        self, question: str, k: int = 5
    ) -> Tuple[Optional[str], float, str]:
        """
        Поиск ответа на вопрос.
        Возвращает (answer, confidence, status).
        status: "answered" | "pending"
        """
        normalized = normalize_question(question)
        cache_key = f"search:{normalized}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        embedding = self.embedding_service.embed(question)
        results = self.faiss_index.search(embedding, k=k)

        if not results:
            self._cache[cache_key] = (None, 0.0, "pending")
            return None, 0.0, "pending"

        best_id, best_score = results[0]
        if best_score < self.similarity_threshold:
            self._cache[cache_key] = (None, best_score, "pending")
            return None, best_score, "pending"

        async with self.session_factory() as session:
            result = await session.execute(
                select(FAQEntry).where(FAQEntry.id == best_id)
            )
            entry = result.scalar_one_or_none()
            if entry:
                self._cache[cache_key] = (entry.answer, best_score, "answered")
                return entry.answer, best_score, "answered"

        self._cache[cache_key] = (None, best_score, "pending")
        return None, best_score, "pending"

    async def get_all_entries(self) -> List[FAQEntry]:
        """Получить все записи FAQ."""
        async with self.session_factory() as session:
            result = await session.execute(select(FAQEntry).order_by(FAQEntry.id))
            return list(result.scalars().all())

    async def get_entry_by_id(self, entry_id: int) -> Optional[FAQEntry]:
        """Получить запись по ID."""
        async with self.session_factory() as session:
            result = await session.execute(select(FAQEntry).where(FAQEntry.id == entry_id))
            return result.scalar_one_or_none()

    def invalidate_cache(self) -> None:
        """Очистить кэш при обновлении FAQ."""
        self._cache.clear()
