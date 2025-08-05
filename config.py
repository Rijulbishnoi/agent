import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    mongodb_url: str = os.getenv("MONGODB_URL", "")
    database_name: str = os.getenv("DATABASE_NAME", "")
    websocket_port: int = 8000
    cors_origins: list[str] = ["*"]
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    
    class Config:
        env_file = ".env"

settings = Settings()
