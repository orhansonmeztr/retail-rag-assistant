from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All runtime configuration, read from environment variables or `.env`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    azure_openai_endpoint: str = ""
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_deployment: str = "gpt-4.1-mini"
    azure_openai_embedding_deployment: str = "text-embedding-3-small"
    # judge model for evaluation; a different model than the agent's avoids self-preference
    judge_deployment: str | None = None
    embedding_dimensions: int = 1536
    openai_timeout_s: float = 30.0
    # retries back off and honour Retry-After on 429/5xx; low-quota deployments need more
    openai_max_retries: int = 6

    azure_search_endpoint: str = ""
    azure_search_api_key: str | None = None
    azure_search_index: str = "retail-knowledge"
    use_semantic_ranker: bool = False

    search_backend: Literal["azure", "memory"] = "azure"
    top_k: int = 5
    max_agent_steps: int = 4
    # extra LLM call that strips claims not supported by the retrieved sources
    answer_guard: bool = True
    warmup_on_startup: bool = True

    # comma-separated keys accepted in X-API-Key; empty disables auth (local development only)
    api_keys: str = ""
    rate_limit_per_minute: int = 30
    # Azure AI Content Safety Prompt Shields; disabled when the endpoint is empty
    content_safety_endpoint: str = ""
    content_safety_api_key: str | None = None

    data_dir: Path = PROJECT_ROOT / "data"
    prompt_path: Path = PROJECT_ROOT / "prompts" / "assistant.yaml"
    golden_path: Path = PROJECT_ROOT / "eval" / "golden.jsonl"


@lru_cache
def get_settings() -> Settings:
    return Settings()
