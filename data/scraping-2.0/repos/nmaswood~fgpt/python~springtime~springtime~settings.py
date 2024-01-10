from pydantic import BaseSettings, Field

from springtime.models.open_ai import OpenAIModel


class Settings(BaseSettings):
    host: str = Field(env="host")
    port: int = Field(env="port")
    openai_api_key: str = Field(env="OPENAI_API_KEY")
    pinecone_api_key: str = Field(env="PINECONE_API_KEY")
    pinecone_env: str = Field(env="PINECONE_ENV")
    pinecone_index: str = Field(env="PINECONE_INDEX")
    pinecone_namespace: str = Field(env="PINECONE_NAMESPACE")
    tracing_enabled: bool = Field(env="TRACING_ENABLED", default=False)
    reload: bool = Field(env="RELOAD", default=False)
    service_to_service_secret: str | None = Field(
        env="SERVICE_TO_SERVICE_SECRET",
    )
    anthropic_api_key: str = Field(env="ANTHROPIC_API_KEY")
    reports_openai_model: OpenAIModel = Field(env="REPORTS_OPENAI_MODEL")
    mock_out_claude: bool = Field(
        env="MOCK_OUT_CLAUDE",
        default=False,
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


SETTINGS = Settings()
