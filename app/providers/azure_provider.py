from __future__ import annotations
import logging
from openai import AsyncAzureOpenAI
from app.config import Settings
from app.providers.base import BaseLLMProvider, BaseEmbedProvider

logger = logging.getLogger("ragdms.providers.azure")


class AzureLLMProvider(BaseLLMProvider):
    def __init__(self, settings: Settings) -> None:
        self._client = AsyncAzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )
        self._deployment = settings.azure_openai_chat_deployment

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ) -> str:
        resp = await self._client.chat.completions.create(
            model=self._deployment,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


class AzureEmbedProvider(BaseEmbedProvider):
    DIMENSION = 1536

    def __init__(self, settings: Settings) -> None:
        self._client = AsyncAzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )
        self._deployment = settings.azure_openai_embed_deployment

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        all_embeddings: list[list[float]] = []
        batch_size = 2048
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = await self._client.embeddings.create(
                model=self._deployment,
                input=batch,
            )
            all_embeddings.extend([item.embedding for item in resp.data])
        return all_embeddings

    @property
    def dimension(self) -> int:
        return self.DIMENSION
