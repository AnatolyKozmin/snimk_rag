"""Нормализация вопросов."""
import re
from typing import Optional


def normalize_question(question: str) -> str:
    """
    Нормализация вопроса для улучшения поиска.
    - Приведение к нижнему регистру
    - Удаление лишних пробелов
    - Удаление пунктуации в конце
    - Базовое исправление опечаток (опционально)
    """
    if not question or not isinstance(question, str):
        return ""

    text = question.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\-?.,!]", "", text)
    text = text.strip()
    return text
