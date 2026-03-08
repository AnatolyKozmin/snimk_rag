"""Сервис для генерации эмбеддингов."""
import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Сервис эмбеддингов на базе sentence-transformers."""

    _instance = None

    def __new__(cls, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Singleton для загрузки модели один раз."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if not hasattr(self, "_model") or self._model is None:
            logger.info("Loading embedding model: %s", model_name)
            self._model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded")

    def embed(self, text: str) -> np.ndarray:
        """Создать эмбеддинг для одного текста."""
        return self._model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Создать эмбеддинги для списка текстов."""
        if not texts:
            return np.array([]).reshape(0, self._model.get_sentence_embedding_dimension())
        return self._model.encode(texts, convert_to_numpy=True)

    @property
    def dimension(self) -> int:
        """Размерность эмбеддингов."""
        return self._model.get_sentence_embedding_dimension()
