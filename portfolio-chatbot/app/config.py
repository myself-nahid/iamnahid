from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Your existing fields...
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.7
    max_tokens: int = 500
    GOOGLE_API_KEY: str
    
    # ADD THIS:
    class Config:
        protected_namespaces = ('settings_',)
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings():
    return Settings()
