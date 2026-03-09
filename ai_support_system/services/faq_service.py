"""Сервис FAQ - поиск ответов по вопросам."""
import asyncio
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
        similarity_threshold_llm: float = 0.6,
        cache_ttl: int = 300,
        cache_max_size: int = 1000,
        llm_service=None,
        use_llm_rag: bool = False,
        llm_top_k: int = 3,
    ):
        self.embedding_service = embedding_service
        self.faiss_index = faiss_index
        self.session_factory = session_factory
        self.similarity_threshold = similarity_threshold
        self.similarity_threshold_llm = similarity_threshold_llm
        self._cache = TTLCache(maxsize=cache_max_size, ttl=cache_ttl)
        self._llm_service = llm_service
        self._use_llm_rag = use_llm_rag and llm_service is not None
        self._llm_top_k = llm_top_k

    async def _get_entries_by_ids(self, ids: List[int]) -> List[FAQEntry]:
        """Получить записи FAQ по списку ID."""
        if not ids:
            return []
        async with self.session_factory() as session:
            result = await session.execute(
                select(FAQEntry).where(FAQEntry.id.in_(ids))
            )
            entries = list(result.scalars().all())
        # Сохранить порядок по ids
        id_to_entry = {e.id: e for e in entries}
        return [id_to_entry[i] for i in ids if i in id_to_entry]

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
        results = self.faiss_index.search(embedding, k=max(k, self._llm_top_k))

        if not results:
            self._cache[cache_key] = (None, 0.0, "pending")
            return None, 0.0, "pending"

        best_id, best_score = results[0]

        # Ниже порога LLM — сразу pending
        if best_score < self.similarity_threshold_llm:
            self._cache[cache_key] = (None, best_score, "pending")
            return None, best_score, "pending"

        # Получаем топ-k записей для контекста (для LLM или fallback)
        top_ids = [r[0] for r in results[: self._llm_top_k]]
        entries = await self._get_entries_by_ids(top_ids)

        if not entries:
            self._cache[cache_key] = (None, best_score, "pending")
            return None, best_score, "pending"

        # Высокая уверенность: ответ есть
        if best_score >= self.similarity_threshold:
            if self._use_llm_rag:
                # RAG: LLM переформулирует ответ
                faq_pairs = [(e.question, e.answer) for e in entries]
                loop = asyncio.get_event_loop()
                try:
                    answer = await loop.run_in_executor(
                        None,
                        lambda: self._llm_service.generate_rag_answer(question, faq_pairs),
                    )
                    self._cache[cache_key] = (answer, best_score, "answered")
                    return answer, best_score, "answered"
                except Exception as e:
                    logger.warning("LLM RAG failed, using raw FAQ: %s", e)
            # Без LLM или при ошибке — сырой ответ
            self._cache[cache_key] = (entries[0].answer, best_score, "answered")
            return entries[0].answer, best_score, "answered"

        # Средняя уверенность (0.6–0.8): пробуем LLM
        if self._use_llm_rag:
            faq_pairs = [(e.question, e.answer) for e in entries]
            loop = asyncio.get_event_loop()
            try:
                answer = await loop.run_in_executor(
                    None,
                    lambda: self._llm_service.generate_rag_answer(question, faq_pairs),
                )
                # Проверяем, не сказал ли LLM "не знаю"
                if "нет ответа" in answer.lower() or "обратитесь к администратору" in answer.lower():
                    self._cache[cache_key] = (None, best_score, "pending")
                    return None, best_score, "pending"
                self._cache[cache_key] = (answer, best_score, "answered")
                return answer, best_score, "answered"
            except Exception as e:
                logger.warning("LLM fallback failed: %s", e)

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
