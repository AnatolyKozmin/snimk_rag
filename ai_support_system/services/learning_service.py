"""Сервис самообучения - добавление новых Q/A в базу."""
import logging
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import FAQEntry, PendingQuestion
from services.embedding_service import EmbeddingService
from services.normalizer import normalize_question
from vectorstore.faiss_index import FAISSIndex

logger = logging.getLogger(__name__)


class LearningService:
    """Сервис обучения - добавление ответов в FAQ."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        faiss_index: FAISSIndex,
        session_factory,
    ):
        self.embedding_service = embedding_service
        self.faiss_index = faiss_index
        self.session_factory = session_factory

    async def add_qa(self, question: str, answer: str) -> FAQEntry:
        """
        Добавить новую пару Q/A в FAQ и обновить FAISS индекс.
        """
        normalized = normalize_question(question)
        embedding = self.embedding_service.embed(question)

        async with self.session_factory() as session:
            entry = FAQEntry(
                question=question,
                answer=answer,
                question_normalized=normalized,
            )
            session.add(entry)
            await session.flush()
            entry_id = entry.id
            await session.commit()

        self.faiss_index.add(embedding, entry_id)
        self.faiss_index.save()
        logger.info("Added FAQ entry: id=%d", entry_id)
        return entry

    async def add_qa_from_pending(self, pending_id: int, answer: str) -> Optional[FAQEntry]:
        """
        Добавить ответ для pending вопроса и пометить его как отвеченный.
        """
        async with self.session_factory() as session:
            result = await session.execute(
                select(PendingQuestion).where(PendingQuestion.id == pending_id)
            )
            pending = result.scalar_one_or_none()
            if not pending:
                return None

            entry = FAQEntry(
                question=pending.question,
                answer=answer,
                question_normalized=pending.question_normalized,
            )
            session.add(entry)
            await session.flush()
            entry_id = entry.id

            pending.answered = True
            pending.faq_entry_id = entry_id
            await session.commit()

        embedding = self.embedding_service.embed(pending.question)
        self.faiss_index.add(embedding, entry_id)
        self.faiss_index.save()
        logger.info("Added FAQ from pending: pending_id=%d, faq_id=%d", pending_id, entry_id)
        return entry

    async def add_qa_for_cluster(
        self, cluster_questions: List[str], answer: str
    ) -> FAQEntry:
        """
        Добавить ответ для кластера похожих вопросов.
        Используется первый вопрос как канонический.
        """
        canonical_question = cluster_questions[0] if cluster_questions else ""
        return await self.add_qa(canonical_question, answer)

    async def mark_cluster_answered(
        self, cluster_id: int, faq_entry_id: int, session: AsyncSession
    ) -> None:
        """Пометить все вопросы кластера как отвеченные."""
        result = await session.execute(
            select(PendingQuestion).where(
                PendingQuestion.cluster_id == cluster_id,
                PendingQuestion.answered == False,
            )
        )
        for pending in result.scalars().all():
            pending.answered = True
            pending.faq_entry_id = faq_entry_id
