"""Configuration management."""

from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "app" / "data"


class Settings(BaseSettings):
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    
    # Paths
    manuals_dir: Path = DATA_DIR / "sample_manuals"
    logs_dir: Path = DATA_DIR / "sample_logs"
    knowledge_dir: Path = DATA_DIR / "knowledge"
    chroma_persist_dir: Path = DATA_DIR / "chroma_db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
