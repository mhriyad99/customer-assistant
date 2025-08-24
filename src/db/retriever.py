from typing import List, Dict, Any
from src.db.vector_db import QdrantVectorStore as VectorStore
from src.db.embedder import OpenAIEmbeddingProvider as EmbeddingProvider


class VectorStoreRetriever:
    def __init__(self, vector_store: VectorStore=VectorStore(),
                 embedding_provider: EmbeddingProvider=EmbeddingProvider()):

        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    def store(self, docs: List[Dict[str, Any]], embedding_field):
        vectors = self.embedding_provider.bulk_embed(docs, embedding_field)
        self.vector_store.add(docs, vectors)

    def query(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        query_vector = self.embedding_provider.embed(query)
        return self.vector_store.search(query_vector, k)


retriever = VectorStoreRetriever()

# docs = [{'text': "you are shit", "id": 3}, {'text': "There is something wrong with you", "id": 4}]
# retriever.store(docs, 'text')
# print(retriever.query("shit"))