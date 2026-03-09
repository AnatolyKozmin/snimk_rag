"""Конфигурация приложения."""
import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Настройки приложения."""

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Telegram
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_ADMIN_IDS: str = ""  # comma-separated user IDs
    API_URL: str = "http://127.0.0.1:8000"  # URL API для бота (в Docker: http://faq_api:8000)

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/faq.db"

    # FAISS
    FAISS_INDEX_PATH: str = "./data/faiss_index.bin"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384  # all-MiniLM-L6-v2

    # LLM RAG (Qwen2.5-1.5B)
    USE_LLM_RAG: bool = True
    LLM_MODEL: str = "Qwen/Qwen2.5-1.5B-Instruct"
    LLM_TOP_K: int = 3  # сколько FAQ передавать в контекст LLM
    MODEL_CACHE_DIR: Path = Path("./model_cache")  # кэш моделей (HF + sentence-transformers)

    # Thresholds
    SIMILARITY_THRESHOLD: float = 0.75
    SIMILARITY_THRESHOLD_LLM: float = 0.6  # мин. score для попытки ответа через LLM
    CLUSTERING_EPS: float = 0.3  # DBSCAN eps
    CLUSTERING_MIN_SAMPLES: int = 2

    # Admin
    ADMIN_SECRET_KEY: str = "change-me-in-production"
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "admin123"

    # Cache
    CACHE_TTL_SECONDS: int = 300
    CACHE_MAX_SIZE: int = 1000

    # Paths
    DATA_DIR: Path = Path("./data")
    LOG_LEVEL: str = "INFO"
    # Файл для автоимпорта при старте (faq.xlsx в корне проекта)
    INITIAL_FAQ_FILE: str = "faq.xlsx"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure data directory exists
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
