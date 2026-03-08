"""Модели базы данных."""
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Базовый класс для моделей."""

    pass


class FAQEntry(Base):
    """Запись в FAQ базе знаний."""

    __tablename__ = "faq_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    question_normalized: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    faiss_index_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class PendingQuestion(Base):
    """Вопрос в очереди администратора."""

    __tablename__ = "pending_questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    question_normalized: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    embedding_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cluster_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    telegram_user_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    answered: Mapped[bool] = mapped_column(Boolean, default=False)
    faq_entry_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("faq_entries.id"), nullable=True
    )


class AnsweredQuestion(Base):
    """История отвеченных вопросов (для аналитики)."""

    __tablename__ = "answered_questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    faq_entry_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("faq_entries.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
