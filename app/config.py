from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    internal_token: str
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "smart_cloud_dms"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    s3_bucket: str = "smart-cloud-dms"

    # Provider: "gemini" (free tier) | "openai" | "azure"
    ai_provider: str = "gemini"

    # Google Gemini (free tier — default)
    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-2.0-flash"
    gemini_embed_model: str = "text-embedding-004"

    # OpenAI (paid)
    openai_api_key: str = ""
    openai_embed_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    # Azure OpenAI (paid)
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-06-01"
    azure_openai_embed_deployment: str = "text-embedding-3-small"
    azure_openai_chat_deployment: str = "gpt-4o-mini"

    # Retrieval
    rag_top_k: int = 10
    rag_similarity_threshold: float = 0.70
    chunk_min_tokens: int = 500
    chunk_max_tokens: int = 800

    # App
    log_level: str = "INFO"
    allowed_origins: str = "*"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
