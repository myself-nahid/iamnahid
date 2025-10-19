from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.7
    max_tokens: int = 500

    model_config = {
        "env_file": ".env",
        "case_sensitive": False
    }


@lru_cache()
def get_settings():
    return Settings()
