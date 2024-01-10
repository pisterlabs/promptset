import os

from dotenv import load_dotenv

from gptif.cl_image import generate_image

load_dotenv()  # take environment variables from .env.

import bugsnag

stage = os.environ.get("STAGE", None)

bugsnag.configure(
    api_key=os.environ["BUGSNAG_API_KEY"] if stage else None,
    project_root=os.getcwd(),
)

import base64
import contextvars
import json
import uuid
from typing import Annotated, Any, Callable, Dict, List, Optional, Tuple, Union, cast

import nacl
import nacl.secret
import nacl.utils
from aws_lambda_powertools.metrics import MetricUnit
from fastapi import APIRouter, Cookie, Depends, FastAPI, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.routing import APIRoute
from jose import jwt
from pydantic import BaseModel
from starlette import status
from starlette.exceptions import ExceptionMiddleware
from starlette.responses import HTMLResponse, Response

import gptif.console
import gptif.handle_input
from gptif.backend_utils import logger, metrics
from gptif.console import ConsoleHandler, session_id_contextvar
from gptif.db import (
    AiImage,
    GameState,
    GptDialogue,
    add_feedback,
    create_db_and_tables,
    get_ai_image_from_id,
    get_ai_image_if_cached,
    get_answer_if_cached,
    get_game_state_from_id,
    push_game_command,
    put_ai_image_in_cache,
    put_answer_in_cache,
    upsert_game_state,
)
from gptif.llm import LlamaCppLanguageModel, OpenAiLanguageModel
from gptif.state import World

root_path = f"/{stage}/" if stage else "/"

app = FastAPI(title="MyAwesomeApp", root_path=root_path)


class LoggerRouteHandler(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def route_handler(request: Request) -> Response:
            # Add fastapi context to logs
            ctx = {
                "path": request.url.path,
                "route": self.path,
                "method": request.method,
            }
            logger.append_keys(fastapi=ctx)
            logger.info("Received request")

            return await original_route_handler(request)

        return route_handler


app.router.route_class = LoggerRouteHandler

origins = [
    "http://gptif-site.s3-website-us-east-1.amazonaws.com",
    "https://gptif-site.s3-website-us-east-1.amazonaws.com",
    "http://generativefiction.com",
    "https://generativefiction.com",
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, err):
    logger.exception("Unhandled exception")
    bugsnag.notify(err)
    metrics.add_metric(name="Crash", unit=MetricUnit.Count, value=1)
    return JSONResponse(
        status_code=500, content={"detail": "Internal Server Error: " + str(err)}
    )


app.add_middleware(ExceptionMiddleware, handlers=app.exception_handlers)

openai_model = OpenAiLanguageModel()

secret_key = base64.b64decode(os.environ["GPTIF_SECRET_KEY"])
secret_box = nacl.secret.SecretBox(secret_key)


class GameCommand(BaseModel):
    command: str


class GameFeedback(BaseModel):
    feedback: str


@app.on_event("startup")
def on_startup():
    create_db_and_tables()


def fetch_session_id(
    session_cookie: Annotated[Union[str, None], Cookie()] = None,
) -> Optional[str]:
    logger.info("FETCHING SESSION ID")
    logger.info(session_cookie)
    logger.info(type(session_cookie))
    if session_cookie is None:
        return None
    session_cookie_bytes = base64.b64decode(session_cookie)
    session_cookie_decrypted = json.loads(
        secret_box.decrypt(session_cookie_bytes).decode()
    )
    logger.info(session_cookie_decrypted)
    session_id = session_cookie_decrypted.get(
        "logged_in_id", session_cookie_decrypted["logged_out_id"]
    )
    logger.info(session_id)
    logger.set_correlation_id(session_id)
    return session_id


@app.post("/api/fetch_dialogue")
async def fetch_dialogue(query: GptDialogue) -> str:
    answer = get_answer_if_cached(query)
    if answer is None and query.model_version == openai_model.model_name():
        # Grab the answer from openai
        assert query.stop_words is not None
        answer = openai_model.llm(
            query.context, stop=query.stop_words.split(","), echo=False
        )
        query.answer = answer
        put_answer_in_cache(query)
        return answer

    return "None" if answer is None else answer


@app.post("/api/fetch_image_id_for_caption")
async def fetch_image_id_for_caption(query: AiImage) -> Optional[str]:
    ai_image = get_ai_image_if_cached(query)
    if ai_image is None:
        image_data_bytes = generate_image(query)

        if image_data_bytes is None:
            bugsnag.notify(Exception(f"Invalid image fetch query: {query.prompt}"))
            return None
        else:
            query.result = image_data_bytes

            put_ai_image_in_cache(query)

            assert query.id is not None

            return str(query.id)
    return str(ai_image.id)


import gzip


@app.get(
    "/api/ai_image/{image_id}",
    responses={200: {"content": {"image/png": {}}}},
    # Prevent FastAPI from adding "application/json" as an additional
    # response media type in the autogenerated OpenAPI specification.
    # https://github.com/tiangolo/fastapi/issues/3258
    response_class=Response,
)
async def ai_image(image_id: str) -> Response:
    int_id = int(image_id)
    ai_image = get_ai_image_from_id(int_id)
    if ai_image is None:
        raise Exception("Oops")
    assert ai_image.result is not None
    return Response(
        content=gzip.compress(ai_image.result),
        media_type="image/png",
        headers={"Content-Encoding": "gzip"},
    )


@app.post("/api/put_dialogue")
async def put_dialogue(query: GptDialogue):
    put_answer_in_cache(query)


@app.get("/")
async def send_to_index():
    return RedirectResponse("index.html")


@app.get("/api")
async def root():
    return {"message": "Hello World!"}


@app.post("/api/begin_game")
async def begin_game() -> JSONResponse:
    session_id = str(uuid.uuid4())
    session_cookie = json.dumps({"logged_out_id": session_id})
    encrypted_session_cookie = base64.b64encode(
        secret_box.encrypt(session_cookie.encode("utf-8"))
    ).decode()
    logger.info(f"SESSION ID {session_id}")
    session_id_contextvar.set(session_id)
    logger.info(gptif.console.console.buffers.get(session_id, []))

    world = World()
    logger.info("NEW GAME")
    game_state = GameState(session_id=session_id)  # type: ignore
    world.start_chapter_one()
    world.save(game_state)
    upsert_game_state(game_state)
    response = JSONResponse(content=gptif.console.console.buffers.get(session_id, []))
    if session_id in gptif.console.console.buffers:
        del gptif.console.console.buffers[session_id]
    session_id_contextvar.set("")
    response.set_cookie("session_cookie", encrypted_session_cookie)
    logger.set_correlation_id(session_id)
    metrics.add_metric(name="StartedGame", unit=MetricUnit.Count, value=1)
    return response


@app.post("/api/handle_input")
async def handle_input(
    command: GameCommand, session_id=Depends(fetch_session_id)
) -> JSONResponse:
    logger.info("IN POST")
    logger.info(session_id)
    if session_id is None:
        raise HTTPException(status_code=400, detail="Sent input but there's no game")
    assert session_id is not None
    logger.info(f"SESSION ID {session_id}")
    session_id_contextvar.set(session_id)
    logger.info(gptif.console.console.buffers.get(session_id, []))

    world = World()
    game_state = get_game_state_from_id(session_id)
    logger.info(f"GAME COMMAND: {command.command}")
    if game_state is None:
        # Game was deleted
        gptif.console.console.print("(Server gamefile missing, starting a new game...)")
        world.start_chapter_one()
        game_state = GameState(session_id=session_id)  # type: ignore
    elif not world.load(game_state):
        gptif.console.console.print(
            "(Incompatible save detected, starting a new game...)"
        )
        world.start_chapter_one()
    else:
        import cProfile, pstats, io
        from pstats import SortKey

        pr = cProfile.Profile()
        pr.enable()
        gptif.handle_input.handle_input(world, command.command)
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.TIME
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(5)
        print(s.getvalue())
        logger.info("COMMAND HANDLED")
        push_game_command(session_id, command.command)
    logger.info(f"SESSION ID {session_id}")
    world.save(game_state)
    logger.info(f"SESSION ID {session_id}")
    upsert_game_state(game_state)
    logger.info(f"SESSION ID {session_id}")
    response = JSONResponse(content=gptif.console.console.buffers.get(session_id, []))
    logger.info("BEFORE AND AFTER")
    logger.info(gptif.console.console.buffers.get(session_id, []))
    if session_id in gptif.console.console.buffers:
        del gptif.console.console.buffers[session_id]
    logger.info(gptif.console.console.buffers.get(session_id, []))
    session_id_contextvar.set("")
    return response


@app.post("/api/feedback")
async def feedback(feedback: GameFeedback, session_id=Depends(fetch_session_id)):
    print(feedback)
    print(feedback.feedback)
    assert feedback.feedback is not None
    add_feedback(feedback.feedback, session_id)
