from typing import Type, Dict, Any, AsyncGenerator
from app.llm.base_llm import BaseLLM
from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GeminiLLM(BaseLLM):
    def __init__(self, model_name: str = "gemini-pro"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.model = ChatGoogleGenerativeAI(
                model=model_name, 
                temperature=0.7, 
                convert_system_message_to_human=True, # Important for older LangChain versions with Gemini
                google_api_key=settings.GEMINI_API_KEY
            )
            logger.info(f"GeminiLLM initialized with model: {model_name}")
        except ImportError:
            logger.error("Please install langchain-google-genai to use GeminiLLM: pip install langchain-google-genai")
            raise

    def get_model(self):
        return self.model

    async def ainvoke(self, prompt: str, **kwargs) -> str:
        response = await self.model.ainvoke(prompt, **kwargs)
        return response.content

    async def astream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        async for chunk in self.model.astream(prompt, **kwargs):
            yield chunk.content if chunk.content else "" # Ensure we always yield a string
            
    def get_token_usage(self, response_metadata: Dict[str, Any]) -> Dict[str, int]:
        # LangChain's astream/ainvoke typically return a Pydantic object
        # The token usage is usually in .response_metadata or similar.
        # This part might need adjustment based on the exact LangChain version's output structure.
        token_usage = response_metadata.get("token_usage", {})
        return {
            "prompt_tokens": token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
            "total_tokens": token_usage.get("total_tokens", 0),
        }


class OllamaLLM(BaseLLM):
    def __init__(self, model_name: str = settings.OLLAMA_MODEL_NAME, base_url: str = settings.OLLAMA_BASE_URL, temperature: float = 0.5, max_retries: int = 3, API_KEY: str = settings.OPENAI_API_KEY):
        try:
            from langchain_openai import ChatOpenAI
            self.model = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_retries=max_retries,
                base_url=base_url,
                api_key=API_KEY,
            )
            logger.info(f"OllamaLLM initialized with model: {model_name} at {base_url}")
        except ImportError:
            logger.error("langchain_openai is not installed. Please install it using pip install langchain_openai")
            raise

    def get_model(self):
        return self.model
    
    async def ainvoke(self, prompt, **kwargs) -> str:
        response = await self.model.ainvoke(prompt, **kwargs)
        return response.content
    
    async def astream(self, prompt, **kwargs) -> AsyncGenerator[str, None]:
        async for chunk in self.model.astream(prompt, **kwargs):
            yield chunk.content if chunk.content else ""

    def get_token_usage(self, response_metadata:Dict[str, Any]) -> Dict[str, int]:
        # Ollama token usage might be in 'llm_output' or similar.
        # This is a placeholder and needs to be adapted based on actual response format.
        llm_output = response_metadata.get("llm_output", {})
        token_usage = llm_output.get("token_usage", {})
        return {
            "prompt_tokens": token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
            "total_tokens": token_usage.get("total_tokens", 0),
        }
    

class LLMFactory:
    _llm_providers: Dict[str, Type[BaseLLM]] = {
        "ollama": OllamaLLM,
        "gemini": GeminiLLM,
        # Add other LLMs here
    }

    @classmethod
    def get_llm(cls, provider_name: str = settings.DEFAULT_LLM_PROVIDER, **kwargs) -> BaseLLM:
        """
        Factory method to get an LLM instance based on the provider name.
        """
        provider_class = cls._llm_providers.get(provider_name)
        if not provider_class:
            raise ValueError(f"Invalid LLM provider: {provider_name}. Valid providers: {list(cls._llm_providers.keys())}")
        logger.info(f"Creating LLM instance for provider: {provider_name} ")
        return provider_class(**kwargs) 
    