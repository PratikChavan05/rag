from __future__ import annotations
import logging
from openai import AsyncOpenAI
from app.config import Settings
from app.providers.base import BaseLLMProvider, BaseEmbedProvider

logger = logging.getLogger("ragdms.providers.openai")


class OpenAILLMProvider(BaseLLMProvider):
    def __init__(self, settings: Settings) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_chat_model

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


class OpenAIEmbedProvider(BaseEmbedProvider):
    DIMENSION = 1536

    def __init__(self, settings: Settings) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_embed_model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # OpenAI supports batch embedding. Process in chunks of 2048 to stay under limits.
        all_embeddings: list[list[float]] = []
        batch_size = 2048
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = await self._client.embeddings.create(
                model=self._model,
                input=batch,
            )
            all_embeddings.extend([item.embedding for item in resp.data])
        return all_embeddings

    @property
    def dimension(self) -> int:
        return self.DIMENSION
