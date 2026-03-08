#!/usr/bin/env python3
"""
Импорт FAQ из Excel (.xlsx) или CSV.
Формат: первая колонка — вопрос, вторая — ответ.
Первая строка может быть заголовком (вопрос/answer и т.п.) — будет пропущена.

Использование:
  python scripts/import_from_excel.py faq.xlsx
  python scripts/import_from_excel.py faq.csv
  python scripts/import_from_excel.py faq.xlsx --clear   # очистить базу перед импортом
"""
import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import settings
from database.db import init_db
from database.models import FAQEntry
from services.embedding_service import EmbeddingService
from services.normalizer import normalize_question
from vectorstore.faiss_index import FAISSIndex


def load_from_file(path: Path) -> list[tuple[str, str]]:
    """Загрузить FAQ из файла (делегирует в services.faq_loader)."""
    from services.faq_loader import load_faq_from_bytes

    return load_faq_from_bytes(path.read_bytes(), path.name)




async def main():
    parser = argparse.ArgumentParser(description="Импорт FAQ из Excel/CSV")
    parser.add_argument("file", type=Path, help="Путь к файлу faq.xlsx или faq.csv")
    parser.add_argument("--clear", action="store_true", help="Очистить базу перед импортом")
    args = parser.parse_args()

    if not args.file.exists():
        print(f"Файл не найден: {args.file}")
        sys.exit(1)

    pairs = load_from_file(args.file)
    if not pairs:
        print("Нет данных для импорта. Проверьте формат: колонка 1 = вопрос, колонка 2 = ответ.")
        sys.exit(1)

    print(f"Загружено {len(pairs)} пар Q/A из {args.file}")

    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    session_factory = await init_db(settings.DATABASE_URL, settings.DATA_DIR)

    if args.clear:
        from sqlalchemy import delete
        async with session_factory() as session:
            await session.execute(delete(FAQEntry))
            await session.commit()
        print("База FAQ очищена.")

    embedding_service = EmbeddingService(settings.EMBEDDING_MODEL)
    faiss_path = Path(settings.FAISS_INDEX_PATH)
    faiss_index = FAISSIndex(
        dimension=embedding_service.dimension,
        index_path=faiss_path,
    )
    if args.clear:
        faiss_index.rebuild(
            np.array([]).reshape(0, embedding_service.dimension), []
        )
    else:
        faiss_index.load()

    questions = [q for q, _ in pairs]
    embeddings = embedding_service.embed_batch(questions)

    async with session_factory() as session:
        for i, (question, answer) in enumerate(pairs):
            entry = FAQEntry(
                question=question,
                answer=answer,
                question_normalized=normalize_question(question),
            )
            session.add(entry)
            await session.flush()
            faiss_index.add(embeddings[i], entry.id)
        await session.commit()

    faiss_index.save()
    print(f"Импортировано {len(pairs)} записей в FAQ. Индекс FAISS обновлён.")


if __name__ == "__main__":
    asyncio.run(main())
