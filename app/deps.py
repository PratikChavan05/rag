from __future__ import annotations
from functools import lru_cache
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.config import get_settings
from app.db.mongo import get_database
from app.providers.base import BaseLLMProvider, BaseEmbedProvider


def get_db() -> AsyncIOMotorDatabase:
    return get_database()


@lru_cache(maxsize=1)
def get_llm_provider() -> BaseLLMProvider:
    settings = get_settings()
    if settings.ai_provider == "gemini":
        from app.providers.gemini_provider import GeminiLLMProvider
        return GeminiLLMProvider(settings)
    if settings.ai_provider == "azure":
        from app.providers.azure_provider import AzureLLMProvider
        return AzureLLMProvider(settings)
    from app.providers.openai_provider import OpenAILLMProvider
    return OpenAILLMProvider(settings)


@lru_cache(maxsize=1)
def get_embed_provider() -> BaseEmbedProvider:
    settings = get_settings()
    if settings.ai_provider == "gemini":
        from app.providers.gemini_provider import GeminiEmbedProvider
        return GeminiEmbedProvider(settings)
    if settings.ai_provider == "azure":
        from app.providers.azure_provider import AzureEmbedProvider
        return AzureEmbedProvider(settings)
    from app.providers.openai_provider import OpenAIEmbedProvider
    return OpenAIEmbedProvider(settings)

