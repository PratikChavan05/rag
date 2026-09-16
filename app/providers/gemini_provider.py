"""Google Gemini provider — free tier for personal projects."""
from __future__ import annotations

import json
import logging

from google import genai
from google.genai import types

from app.config import Settings
from app.providers.base import BaseEmbedProvider, BaseLLMProvider

logger = logging.getLogger("ragdms.providers.gemini")


class GeminiLLMProvider(BaseLLMProvider):
    """Chat completions via Gemini (gemini-2.0-flash on free tier)."""

    def __init__(self, settings: Settings) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_chat_model

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ) -> str:
        # Convert OpenAI-style messages to Gemini format
        contents: list[types.Content] = []
        system_instruction: str | None = None

        for msg in messages:
            role = msg["role"]
            text = msg["content"]
            if role == "system":
                system_instruction = text
            elif role == "user":
                contents.append(
                    types.Content(role="user", parts=[types.Part(text=text)])
                )
            elif role == "assistant":
                contents.append(
                    types.Content(role="model", parts=[types.Part(text=text)])
                )

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )
        return response.text or ""


class GeminiEmbedProvider(BaseEmbedProvider):
    """Embeddings via Gemini (text-embedding-004, 768-dim on free tier)."""

    DIMENSION = 768

    def __init__(self, settings: Settings) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_embed_model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        all_embeddings: list[list[float]] = []
        # Gemini supports batch of up to 100 texts
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = await self._client.aio.models.embed_content(
                model=self._model,
                contents=batch,
            )
            all_embeddings.extend(
                [emb.values for emb in result.embeddings]
            )
        return all_embeddings

    @property
    def dimension(self) -> int:
        return self.DIMENSION
