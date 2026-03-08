"""FAISS векторный индекс."""
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FAISSIndex:
    """Обёртка над FAISS индексом."""

    def __init__(self, dimension: int, index_path: Optional[Path] = None):
        self.dimension = dimension
        self.index_path = index_path
        self._index: Optional[faiss.IndexFlatIP] = None
        self._id_mapping: List[int] = []  # faiss_idx -> faq_entry_id

    def _create_index(self) -> faiss.IndexFlatIP:
        """Создать новый индекс (cosine similarity через inner product после нормализации)."""
        return faiss.IndexFlatIP(self.dimension)

    def load(self) -> bool:
        """Загрузить индекс с диска."""
        if not self.index_path or not self.index_path.exists():
            logger.info("FAISS index not found, creating new")
            self._index = self._create_index()
            self._id_mapping = []
            return False

        try:
            self._index = faiss.read_index(str(self.index_path))
            mapping_path = self.index_path.with_suffix(".mapping")
            if mapping_path.exists():
                self._id_mapping = [
                    int(x) for x in mapping_path.read_text().strip().split("\n") if x
                ]
            else:
                self._id_mapping = list(range(self._index.ntotal))
            logger.info("FAISS index loaded: %d vectors", self._index.ntotal)
            return True
        except Exception as e:
            logger.warning("Failed to load FAISS index: %s", e)
            self._index = self._create_index()
            self._id_mapping = []
            return False

    def save(self) -> None:
        """Сохранить индекс на диск."""
        if not self._index or self.index_path is None:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        mapping_path = self.index_path.with_suffix(".mapping")
        mapping_path.write_text("\n".join(str(x) for x in self._id_mapping))
        logger.info("FAISS index saved: %d vectors", self._index.ntotal)

    def add(self, embedding: np.ndarray, faq_entry_id: int) -> None:
        """Добавить вектор в индекс (нормализация для cosine similarity)."""
        if self._index is None:
            self._index = self._create_index()
        embedding = embedding.astype(np.float32)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        faiss.normalize_L2(embedding)
        self._index.add(embedding)
        self._id_mapping.append(faq_entry_id)

    def add_batch(self, embeddings: np.ndarray, faq_entry_ids: List[int]) -> None:
        """Добавить несколько векторов."""
        if self._index is None:
            self._index = self._create_index()
        embeddings = embeddings.astype(np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        faiss.normalize_L2(embeddings)
        self._index.add(embeddings)
        self._id_mapping.extend(faq_entry_ids)

    def search(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Поиск k ближайших соседей.
        Возвращает список (faq_entry_id, similarity).
        Используем IndexFlatIP - inner product. Для cosine нужно нормализовать векторы.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        query = query_embedding.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        # Нормализация для cosine similarity
        faiss.normalize_L2(query)
        distances, indices = self._index.search(query, min(k, self._index.ntotal))

        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(self._id_mapping):
                # Inner product после L2 нормализации = cosine similarity
                results.append((self._id_mapping[idx], float(dist)))
        return results

    def rebuild(self, embeddings: np.ndarray, faq_entry_ids: List[int]) -> None:
        """Полная пересборка индекса."""
        self._index = self._create_index()
        self._id_mapping = []
        if len(embeddings) > 0:
            embeddings = embeddings.astype(np.float32)
            faiss.normalize_L2(embeddings)
            self._index.add(embeddings)
            self._id_mapping = list(faq_entry_ids)
        logger.info("FAISS index rebuilt: %d vectors", len(faq_entry_ids))

    @property
    def ntotal(self) -> int:
        """Количество векторов в индексе."""
        return self._index.ntotal if self._index else 0
