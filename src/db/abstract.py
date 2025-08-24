from abc import ABC, abstractmethod
from typing import List, Dict, Any


class EmbeddingProvider(ABC):
    @abstractmethod
    def bulk_embed(self, texts: List[Dict[str, Any]], embedding_field) -> List[List[float]]:
        pass

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        pass


class VectorStore(ABC):
    @abstractmethod
    def add(self, docs: List[Dict[str, Any]], embeddings: List[List[float]]):
        pass

    @abstractmethod
    def search(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        pass