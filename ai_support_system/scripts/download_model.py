#!/usr/bin/env python3
"""
Ручная загрузка модели эмбеддингов.
Запустите перед первым docker compose up — модель сохранится в ./model_cache
и будет подхвачена контейнером (без загрузки при старте).

Использование:
  python scripts/download_model.py

  # С токеном Hugging Face (быстрее):
  HF_TOKEN=hf_xxx python scripts/download_model.py
"""
import os
import sys
from pathlib import Path

# Путь к кэшу — тот же, что монтируется в Docker
CACHE_DIR = Path(__file__).parent.parent / "model_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Hugging Face использует эти переменные
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR / "transformers")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(CACHE_DIR)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    print(f"Загрузка модели {MODEL_NAME}")
    print(f"Кэш: {CACHE_DIR.absolute()}")
    print("—" * 50)

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODEL_NAME)
        # Проверка
        _ = model.encode("test")
        print("—" * 50)
        print("Готово. Модель загружена и сохранена.")
        print(f"Размер кэша: {sum(f.stat().st_size for f in CACHE_DIR.rglob('*') if f.is_file()) / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
