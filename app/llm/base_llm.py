from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any

class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def get_model(self):
        """Returns the LLM model instance."""
        pass

    @abstractmethod
    async def ainvoke(self, prompt: str, **kwargs) -> str:
        """Asynchronously invokes the LLM with a given prompt."""
        pass

    @abstractmethod
    async def astream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Asynchronously streams response from the LLM."""
        yield # Required for abstract async generator

    @abstractmethod
    def get_token_usage(self, response_metadata: Dict[str, Any]) -> Dict[str, int]:
        """Extracts token usage information from the LLM response metadata."""
        pass

