from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime configuration sourced from environment variables.

    LLM runs locally via vLLM (no paid cloud API required).
    """

    # Local Ollama OpenAI-compatible endpoint
    vllm_url: str = "http://host.docker.internal:11434/v1"

    # Qdrant vector DB
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    clinical_collection: str = "clinical_protocols"

    class Config:
        env_prefix = ""
        case_sensitive = False


settings = Settings()
