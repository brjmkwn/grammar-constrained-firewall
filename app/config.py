from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_NAME: str = "grammar-constrained-firewall"
    APP_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    MODEL_ID: str = "qwen-2.5-7b-instruct"
    TOKENIZER_NAME: str = "gpt-4o"
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_MAX_TOKENS: int = 2048

    FSM_CACHE_SIZE: int = 256
    ENABLE_WHITESPACE_FLEXIBILITY: bool = True
    MAX_SCHEMA_DEPTH: int = 10

    ENABLE_METRICS: bool = True
    MAX_LATENCY_HISTORY: int = 10000


settings = Settings()
