from app.auth.adapters.openai import OpenAi
from pydantic import BaseSettings

from app.config import database

from .adapters.jwt_service import JwtService
from .repository.repository import AuthRepository
from .adapters.s3_service import S3Service


class AuthConfig(BaseSettings):
    JWT_ALG: str = "HS256"
    JWT_SECRET: str = "YOUR_SUPER_SECRET_STRING"
    JWT_EXP: int = 10_800


config = AuthConfig()

class Config(BaseSettings):
    OPENAI_API_KEY:str
    AWS_ACCESS_KEY_ID:str
    AWS_SECRET_ACCESS_KEY:str


class Service:
    def __init__(
        self,
        repository: AuthRepository,
        jwt_svc: JwtService,
    ):
        openai = Config()
        self.repository = repository
        self.jwt_svc = jwt_svc
        self.s3_service = S3Service(openai.AWS_ACCESS_KEY_ID, openai.AWS_SECRET_ACCESS_KEY)
        self.openai = OpenAi(openai.OPENAI_API_KEY)


def get_service():
    repository = AuthRepository(database)
    jwt_svc = JwtService(config.JWT_ALG, config.JWT_SECRET, config.JWT_EXP)

    svc = Service(repository, jwt_svc)
    return svc
