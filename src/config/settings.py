import pathlib
from pydantic_settings import BaseSettings

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    API_KEY: str
    EMBEDDING_MODEL: str
    PROMPT_LIMIT: int
    CHAT_MODEL: str
    VECTORDB_URI: str
    VECTOR_COLLECTION_NAME: str
    LOG_LEVEL: int

    class Config:
        env_file = PROJECT_ROOT / ".env"

settings = Settings()
