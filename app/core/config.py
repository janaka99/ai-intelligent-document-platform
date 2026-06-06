from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    # App
    app_name: str = "Agentic Starter"
    app_env: str = "development"
    app_version: str = "0.1.0"
    log_level: str = "INFO"

    # LLM
    openai_api_key: str 

    
    # Automatically read from a .env file
    model_config = SettingsConfigDict(
        env_file= ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    
@lru_cache
def get_settings() -> Settings:
    return Settings()