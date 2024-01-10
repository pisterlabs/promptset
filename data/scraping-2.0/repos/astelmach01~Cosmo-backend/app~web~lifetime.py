import logging
from typing import Awaitable, Callable

import openai
from aiohttp import ClientSession
from fastapi import FastAPI

from app.services.aws.rds import DatabaseSession

DB_NAME = "user_task_db"


def register_startup_event(app: FastAPI) -> Callable[[], Awaitable[None]]:
    @app.on_event("startup")
    async def _startup() -> None:
        app.middleware_stack = None
        DatabaseSession.initialize()
        openai.aiosession.set(ClientSession())

        app.state.db_session = DatabaseSession.get_session()
        app.middleware_stack = app.build_middleware_stack()

    return _startup


def register_shutdown_event(app: FastAPI) -> Callable[[], Awaitable[None]]:
    @app.on_event("shutdown")
    async def _shutdown() -> None:
        DatabaseSession.close(app.state.db_session)

        if session := openai.aiosession.get():
            await session.close()
        else:
            logging.warning("OpenAI session not initialized")

    return _shutdown
