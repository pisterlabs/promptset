import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api import router
from api.home.home import home_router

from app.exceptions import CustomException

from dotenv import load_dotenv

load_dotenv('.env.example')

app_env = os.getenv("ENV", "dev")

import os
import openai

def init_openai_config() -> None:
    api_key = os.getenv("OPENAI_API_KEY", "")
    api_base = os.getenv("OPENAI_API_BASE", "")

    if (api_key == "") or (api_base == ""):
        raise Exception('Sorry, need api key and openai api_base!')
    openai.api_base = api_base
    openai.api_key = api_key


def init_routers(app_: FastAPI) -> None:
    app_.include_router(home_router)
    app_.include_router(router)


def init_listeners(app_: FastAPI) -> None:
    # Exception handler
    @app_.exception_handler(CustomException)
    async def custom_exception_handler(request: Request, exc: CustomException):
        return JSONResponse(
            status_code=exc.code,
            content={"error_code": exc.error_code, "message": exc.message},
        )



def create_app() -> FastAPI:
    app_ = FastAPI(
        title="Knowledgeable QA Svc",
        description="Record Knowledge's embedding and then answer the questions.",
        version="1.0.0",
        docs_url=None if app_env == "prod" else "/docs",
        redoc_url=None if app_env == "prod" else "/redoc"
    )
    init_routers(app_=app_)
    init_listeners(app_=app_)
    init_openai_config()
    return app_


app = create_app()
