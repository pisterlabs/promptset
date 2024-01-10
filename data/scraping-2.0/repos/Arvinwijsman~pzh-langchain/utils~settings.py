import openai
from pydantic import BaseSettings


class Settings(BaseSettings):
    # APP
    openai_api_key: str
    openai_model: str = "gpt-3.5-turbo"
    openai_temperature: float = 0
    redis_host: str = "localhost"
    redis_port: int = 6379
    documents_path: str = "./documents/"
    chunk_size: int = 1500
    chunk_overlap: int = 100

    # Webpage
    home_title: str = "PZH - LAZYTPOD"

    @property
    def redis_dsn(self):
        return f"redis://{self.redis_host}:{self.redis_port}"

    class Config:
        env_file = ".env"


settings = Settings()
