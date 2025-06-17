from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="./.env", extra="ignore")

    PROJECT_NAME: str = "ai-agent"
    API_V1_STR: str = "/api/v1"

    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./sql_app.db"  # For SQLite
    # DATABASE_URL: str = "postgresql+asyncpg://user:password@host:port/dbname" # For PostgreSQL

    # ChromaDB
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_data" # Local persistence

    # LLM Providers
    DEFAULT_LLM_PROVIDER: str = "ollama"
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "") # If using OpenAI via LangChain
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL_NAME: str = os.getenv("OLLAMA_MODEL_NAME", "qwq:32b") # Example Ollama model

    # Agent specific 
    AGENT_MAX_ITERATIONS: int = 10

settings = Settings()