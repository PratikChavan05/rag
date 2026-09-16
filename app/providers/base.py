from __future__ import annotations
from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """Abstract LLM provider for chat completions."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ) -> str:
        """Send a chat completion request and return the assistant's response text."""
        ...


class BaseEmbedProvider(ABC):
    """Abstract embedding provider."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of embedding vectors."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension."""
        ...
