"""Сервис для генерации эмбеддингов."""
import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Размерность для all-MiniLM-L6-v2 (без загрузки модели)
DEFAULT_DIMENSION = 384


class EmbeddingService:
    """Сервис эмбеддингов на базе sentence-transformers. Ленивая загрузка модели."""

    _instance = None

    def __new__(cls, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None

    def _ensure_loaded(self):
        """Загрузить модель при первом обращении."""
        if self._model is None:
            

            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding model loaded")

    def embed(self, text: str) -> np.ndarray:
        """Создать эмбеддинг для одного текста."""
        self._ensure_loaded()
        return self._model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Создать эмбеддинги для списка текстов."""
        self._ensure_loaded()
        if not texts:
            return np.array([]).reshape(0, self._model.get_sentence_embedding_dimension())
        return self._model.encode(texts, convert_to_numpy=True)

    @property
    def dimension(self) -> int:
        """Размерность эмбеддингов (без загрузки модели)."""
        return DEFAULT_DIMENSION

    @property
    def is_loaded(self) -> bool:
        """Модель загружена."""
        return self._model is not None
