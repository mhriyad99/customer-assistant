import uuid
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from src.db.abstract import VectorStore
from src.config.settings import settings

class QdrantVectorStore(VectorStore):
    def __init__(self, qdrant_url: str = settings.VECTORDB_URI,
                 collection_name: str = settings.VECTOR_COLLECTION_NAME,
                 vector_size: int = 1536):

        self.qdrant = QdrantClient(url=qdrant_url)
        self.collection = collection_name

        if not self.qdrant.collection_exists(self.collection):
            self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def add(self, docs: List[Dict[str, Any]], embeddings: List[List[float]]):
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload=doc,
            )
            for doc, vec in zip(docs, embeddings)
        ]
        self.qdrant.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        results = self.qdrant.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=k,
        )
        return [{**res.payload, "similarity": res.score} for res in results]
