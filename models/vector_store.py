"""
RAM cache embeddings + cosine similarity search (NumPy)
"""

import logging
from typing import List, Dict, Optional, Tuple

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SIMILARITY_THRESHOLD, TOP_K

logger = logging.getLogger(__name__)

# (id, ho_ten, similarity)
SearchResult = Tuple[int, str, float]


class VectorStore:
    """
    Lưu toàn bộ embedding trong RAM để tìm kiếm nhanh bằng cosine similarity.
    Thread-safe (chỉ đọc sau khi load).
    """

    _instance: Optional["VectorStore"] = None

    @classmethod
    def instance(cls) -> "VectorStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._ids: List[int]       = []
        self._names: List[str]     = []
        self._matrix: Optional[np.ndarray] = None   # shape (N, 512)
        self._n = 0

    #  Load / Reload    

    def load_from_db(self, db_manager) -> int:
        """
        Nạp tất cả embedding từ DB vào RAM.
        Trả về số người đã load.
        """
        records = db_manager.get_all_embeddings()
        if not records:
            self._ids = []
            self._names = []
            self._matrix = None
            self._n = 0
            logger.info("VectorStore: 0 embeddings loaded.")
            return 0

        ids, names, vecs = [], [], []
        for r in records:
            ids.append(r["id"])
            names.append(r["ho_ten"])
            vecs.append(r["vec"])

        self._ids   = ids
        self._names = names
        self._matrix = np.stack(vecs, axis=0).astype(np.float32)   # (N, D)
        self._n = len(ids)

        # Đảm bảo đã normalize L2 (InsightFace trả về normed_embedding nhưng để chắc)
        norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._matrix = self._matrix / norms

        logger.info(f"VectorStore: {self._n} embeddings loaded.")
        return self._n

    #  Search    

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> List[SearchResult]:
        """
        Tìm top_k người có embedding gần nhất với query_vec.
        Trả về list (person_id, ho_ten, similarity) đã lọc theo threshold.
        """
        if self._matrix is None or self._n == 0:
            return []

        # Normalize query
        q = query_vec.astype(np.float32)
        norm = np.linalg.norm(q)
        if norm == 0:
            return []
        q = q / norm

        # Cosine similarity = dot product (vì đã normalize cả hai)
        sims = self._matrix @ q          # shape (N,)
        top_k = min(top_k, self._n)
        top_indices = np.argpartition(sims, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

        results: List[SearchResult] = []
        for idx in top_indices:
            sim = float(sims[idx])
            if sim >= threshold:
                results.append((self._ids[idx], self._names[idx], sim))
        return results

    def search_best(
        self,
        query_vec: np.ndarray,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> Optional[SearchResult]:
        """Trả về kết quả tốt nhất duy nhất hoặc None."""
        results = self.search(query_vec, top_k=1, threshold=threshold)
        return results[0] if results else None

    #  Thêm nhanh không reload 
    def add(self, person_id: int, ho_ten: str, vec: np.ndarray):
        """Thêm 1 embedding vào cache mà không cần reload từ DB."""
        v = vec.astype(np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        self._ids.append(person_id)
        self._names.append(ho_ten)
        if self._matrix is None:
            self._matrix = v.reshape(1, -1)
        else:
            self._matrix = np.vstack([self._matrix, v.reshape(1, -1)])
        self._n += 1

    @property
    def size(self) -> int:
        return self._n
