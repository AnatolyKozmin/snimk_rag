"""Сервис для работы с очередью вопросов администратора."""
import logging
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import PendingQuestion

logger = logging.getLogger(__name__)


class PendingService:
    """Сервис очереди pending вопросов."""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def add(
        self,
        question: str,
        question_normalized: Optional[str] = None,
        telegram_user_id: Optional[int] = None,
        cluster_id: Optional[int] = None,
    ) -> PendingQuestion:
        """Добавить вопрос в очередь."""
        async with self.session_factory() as session:
            pending = PendingQuestion(
                question=question,
                question_normalized=question_normalized,
                telegram_user_id=telegram_user_id,
                cluster_id=cluster_id,
            )
            session.add(pending)
            await session.commit()
            await session.refresh(pending)
            logger.info("Added pending question: id=%d", pending.id)
            return pending

    async def get_pending(
        self, include_answered: bool = False
    ) -> List[PendingQuestion]:
        """Получить все pending вопросы."""
        async with self.session_factory() as session:
            q = select(PendingQuestion).order_by(PendingQuestion.created_at.desc())
            if not include_answered:
                q = q.where(PendingQuestion.answered == False)
            result = await session.execute(q)
            return list(result.scalars().all())

    async def get_by_id(self, pending_id: int) -> Optional[PendingQuestion]:
        """Получить pending по ID."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(PendingQuestion).where(PendingQuestion.id == pending_id)
            )
            return result.scalar_one_or_none()
