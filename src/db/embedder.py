import openai
from typing import List, Dict, Any
from src.db.abstract import EmbeddingProvider
from src.config.settings import settings


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, oai_client=openai.Client(api_key=settings.API_KEY),
                 model: str = settings.EMBEDDING_MODEL,
                 batch_size: int = 100):
        self.client = oai_client
        self.model = model
        self.batch_size = batch_size

    def bulk_embed(self, docs: List[Dict[str, Any]], embedding_field) -> List[List[float]]:
        texts = [doc[embedding_field] for doc in docs]
        vectors = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            vectors.extend([e.embedding for e in resp.data])

        return vectors

    def embed(self, text: str) -> List[float]:
        embeddings = self.client.embeddings.create(model=self.model, input=text)
        return embeddings.data[0].embedding