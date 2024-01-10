from os import getenv

import redis.asyncio as redis
from databases import Database

from shared.db import PgRepository, create_db_string
from shared.entities import (
    Attempt,
    Block,
    Quiz,
    QuizComplexity,
    User,
    AttemptStat,
    QuizInfo,
)
from shared.redis import ContainerRepository
from shared.resources import SharedResources
from shared.utils import SHARED_CONFIG_PATH
from docker import DockerClient

from openai import AsyncOpenAI


class Context:
    def __init__(self):
        self.shared_settings = SharedResources(f"{SHARED_CONFIG_PATH}/settings.json")
        self.pg = Database(create_db_string(self.shared_settings.pg_creds))
        self.user_repo = PgRepository(self.pg, User)
        self.quiz_repo = PgRepository(self.pg, Quiz)
        self.block_repo = PgRepository(self.pg, Block)
        self.attempt_repo = PgRepository(self.pg, Attempt)
        self.complexity_repo = PgRepository(self.pg, QuizComplexity)
        self.stats_repo = PgRepository(self.pg, AttemptStat)
        self.quiz_info_repo = PgRepository(self.pg, QuizInfo)
        self.docker_pool = [
            [0, DockerClient(**host.model_dump())]
            for host in self.shared_settings.docker_settings.docker_hosts
        ]

        redis_creds = self.shared_settings.redis_creds
        self.redis = redis.Redis(
            host=redis_creds.host,
            port=redis_creds.port,
            # username=redis_creds.username,
            # password=redis_creds.password,
            decode_responses=True,
        )

        self.openai_client = AsyncOpenAI(
            api_key=self.shared_settings.openai_key,
        )

        self.container_repo = ContainerRepository(self.redis, "containers")

        self.access_token_expire_minutes = int(
            getenv("ACCESS_TOKEN_EXPIRE_MINUTES") or 2 * 24 * 60
        )
        self.refresh_token_expire_minutes = int(
            getenv("REFRESH_TOKEN_EXPIRE_MINUTES") or 100 * 24 * 60
        )
        self.jwt_secret_key = getenv("JWT_SECRET_KEY") or "secret"
        self.jwt_refresh_secret_key = getenv("JWT_SECRET_KEY") or "secret"
        self.hash_algorithm = getenv("ALGORITHM") or "HS256"

    async def init_db(self) -> None:
        await self.pg.connect()

    async def dispose_db(self) -> None:
        await self.pg.disconnect()


ctx = Context()
