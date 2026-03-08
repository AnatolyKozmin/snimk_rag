"""Сервис кластеризации похожих вопросов."""
import logging
from typing import List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


class ClusteringService:
    """Кластеризация вопросов по эмбеддингам."""

    def __init__(self, eps: float = 0.3, min_samples: int = 2):
        self.eps = eps
        self.min_samples = min_samples

    def cluster(
        self, embeddings: np.ndarray, questions: List[str]
    ) -> List[Tuple[int, List[Tuple[int, str]]]]:
        """
        Кластеризация по эмбеддингам.
        Возвращает список кластеров: [(cluster_id, [(question_idx, question_text), ...]), ...]
        cluster_id = -1 означает шум (отдельный кластер для каждого).
        """
        if len(embeddings) == 0:
            return []

        embeddings = embeddings.astype(np.float32)
        # Нормализация для cosine distance
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings_norm = embeddings / norms
        # DBSCAN с metric='cosine' или используем предвычисленную матрицу
        # metric='cosine' в sklearn означает 1 - cosine_similarity
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="cosine")
        labels = clustering.fit_predict(embeddings_norm)

        clusters: dict = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append((idx, questions[idx] if idx < len(questions) else ""))

        result = []
        for label, items in sorted(clusters.items()):
            result.append((int(label), items))
        return result
