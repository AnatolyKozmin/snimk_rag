#!/usr/bin/env python3
"""
Скрипт для добавления начальных FAQ записей.
Использование: python scripts/seed_faq.py
"""
import asyncio
import sys
from pathlib import Path

# Добавляем корень проекта в path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import settings
from database.db import init_db
from database.models import FAQEntry
from services.embedding_service import EmbeddingService
from services.normalizer import normalize_question
from vectorstore.faiss_index import FAISSIndex


INITIAL_FAQ = [
    ("Как оплатить заказ?", "Вы можете оплатить заказ банковской картой при оформлении или наличными при получении."),
    ("Какие способы доставки?", "Доставка осуществляется курьером или в пункты выдачи заказов. Сроки зависят от региона."),
    ("Как отследить заказ?", "Отследить заказ можно в личном кабинете по номеру заказа или через SMS-уведомления."),
    ("Как вернуть товар?", "Возврат возможен в течение 14 дней. Оформите заявку в личном кабинете или обратитесь в поддержку."),
    ("Как связаться с поддержкой?", "Напишите в чат на сайте, отправьте email на support@example.com или позвоните по телефону горячей линии."),
]


async def main():
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    session_factory = await init_db(settings.DATABASE_URL, settings.DATA_DIR)

    embedding_service = EmbeddingService(settings.EMBEDDING_MODEL)
    faiss_path = Path(settings.FAISS_INDEX_PATH)
    faiss_index = FAISSIndex(
        dimension=embedding_service.dimension,
        index_path=faiss_path,
    )
    faiss_index.load()

    questions = [q for q, _ in INITIAL_FAQ]
    embeddings = embedding_service.embed_batch(questions)

    async with session_factory() as session:
        for i, (question, answer) in enumerate(INITIAL_FAQ):
            entry = FAQEntry(
                question=question,
                answer=answer,
                question_normalized=normalize_question(question),
            )
            session.add(entry)
            await session.flush()
            emb = embeddings[i]
            faiss_index.add(emb, entry.id)
        await session.commit()

    faiss_index.save()
    print(f"Добавлено {len(INITIAL_FAQ)} записей в FAQ")


if __name__ == "__main__":
    asyncio.run(main())
