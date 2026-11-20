"""All settings come from environment variables (or a .env file).

Every model and provider name lives here so experiments are a config change,
not a code change.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    database_url: str = "postgresql://rag:rag@localhost:5432/rag"

    # Embeddings. "fake" is a deterministic hashing embedder for tests/CI only.
    embedding_provider: Literal["openai", "local", "fake"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    embedding_batch_size: int = 64

    # Generation
    # "openai" means any OpenAI-compatible API: set llm_base_url for Ollama, Gemini, Groq, Mistral...
    llm_provider: Literal["anthropic", "openai", "fake"] = "anthropic"
    llm_model: str = "claude-haiku-5-5"
    llm_base_url: str | None = None
    llm_api_key: str | None = None  # key for llm_base_url; falls back to OPENAI_API_KEY
    judge_model: str | None = None  # LLM-as-judge model; ideally stronger. Defaults to llm_model
    llm_max_tokens: int = 700
    llm_temperature: float = 0.0

    # Reranking
    reranker: Literal["cohere", "local", "none"] = "none"
    cohere_rerank_model: str = "rerank-v3.5"
    local_rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Chunking (tokens are approximated as chars / 4)
    chunk_tokens: int = 500
    chunk_overlap_tokens: int = 50
    section_min_tokens: int = 40

    # Retrieval
    candidates: int = 40  # pool size fed to RRF / reranker
    top_k: int = 5  # chunks given to the LLM
    rrf_k: int = 60

    # Optional cost accounting, USD per 1M tokens. Leave 0 to report tokens only.
    price_llm_input: float = 0.0
    price_llm_output: float = 0.0
    price_embedding: float = 0.0

    # API keys (read by the SDKs too, declared here so .env works)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    cohere_api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
