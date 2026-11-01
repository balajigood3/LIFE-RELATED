from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache()
def get_settings():
    return Settings()
