import os
import typing

import httpx
import jinja_partials
import openai
from fastapi import Request
from fastapi.templating import Jinja2Templates

openai.api_key = os.environ.get("OPENAI_API_KEY")


def time_cache(seconds_to_cache):
    def give_time_parameter(fn):
        def new_fn(time_param, *args, **kwargs):
            return fn(*args, **kwargs)

        return new_fn

    def time_cache_real(fn):
        import time
        from functools import lru_cache

        cached_with_time_param = lru_cache(5)(give_time_parameter(fn))

        def provide_time(*args, **kwargs):
            return cached_with_time_param(
                time.time() // seconds_to_cache, *args, **kwargs
            )

        return provide_time

    return time_cache_real


def get_llm_response(system: str, user: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )

    return response["choices"][0]["message"]["content"]  # type: ignore


def get_sub_from_jwt(session_user):
    return session_user["userinfo"]["sub"]


def flash(request: Request, message: typing.Any, category: str = "primary") -> None:
    if "_messages" not in request.session:
        request.session["_messages"] = []
        request.session["_messages"].append({"message": message, "category": category})


def get_flashed_messages(request: Request):
    return request.session.pop("_messages") if "_messages" in request.session else []


def write_notification(message=""):
    httpx.post(f"https://ntfy.sh/thornewolf", json={"message": message})


templates = Jinja2Templates(directory="app/templates")
templates.env.globals["get_flashed_messages"] = get_flashed_messages
templates.env.globals["ENV"] = os.environ.get("ENV")
templates.env.globals["ENABLE_TRACKING"] = os.environ.get("ENABLE_TRACKING")
templates.env.globals["ENABLE_LIT"] = os.environ.get("ENABLE_LIT")
jinja_partials.register_starlette_extensions(templates)
