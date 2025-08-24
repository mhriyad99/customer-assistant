import openai
from typing import List, Dict, Any
from src.db.abstract import EmbeddingProvider
from src.config.settings import settings


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, oai_client=openai.Client(api_key=settings.API_KEY),
                 model: str = settings.EMBEDDING_MODEL):
        self.client = oai_client
        self.model = model

    def bulk_embed(self, docs: List[Dict[str, Any]], embedding_field) -> List[List[float]]:
        embeddings = self.client.embeddings.create(model=self.model, input=[doc[embedding_field] for doc in docs])
        return [e.embedding for e in embeddings.data]

    def embed(self, text: str) -> List[float]:
        embeddings = self.client.embeddings.create(model=self.model, input=text)
        return embeddings.data[0].embedding