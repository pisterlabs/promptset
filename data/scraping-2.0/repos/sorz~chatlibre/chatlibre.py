#!/usr/bin/env python3
import os
import csv
import json
import socket
import logging
from functools import cache
from pathlib import Path
from typing import Dict, Any, List

import openai
from openai import AsyncOpenAI
from aiohttp import web


MODELS = ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"]
MODELS_USE_JSON_MODE = True
PROMPT = """You are a translation service for fediverse posts. Given JSON \
array of text, detect its language and translate each text to <TARGET>. Keep \
HTML tags, emoji codes (e.g., :smile:), and emoticons intact. Provide the \
results in the following JSON format:

{
  "detectedLanguage": {
    "confidence": 87,
    "language": "zh"
  },
  "translatedText": [
    "<p>Hello!</p>",
    "Bye"
  ]
}"""


routes = web.RouteTableDef()
openai_app_key = web.AppKey("openai_key", AsyncOpenAI)


@cache
def languages_code_name() -> Dict[str, str]:
    with open('iso_639_1.csv', newline='') as f:
        reader = csv.reader(f, delimiter=',')
        return { code: name for code, name, _ in reader }


def generate_supported_languages():
    langs = []
    codes = []
    with open('iso_639_1.csv', newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for code, name, _ in reader:
            codes.append(code)
            langs.append(dict(code=code, name=name, targets=codes))
    return langs


@routes.get('/')
async def index(_: web.Request) -> web.Response:
    return web.Response(text="It's running!")


@routes.get('/languages')
async def languages(_: web.Request) -> web.Response:
    code_name = languages_code_name()
    targets = list(code_name.keys())
    langs = [dict(code=code, name=name, targets=targets)
             for code, name in code_name.items()]
    return web.json_response(langs)


async def chat(
    client: AsyncOpenAI,
    text: List[str] | str,
    target_code: str,
    model: str
) -> Dict[str, Any]:
    if isinstance(text, str):
        text_list = [text]
    else:
        text_list = text
    target = languages_code_name().get(target_code, target_code)
    kwargs: Dict[str, Any] = dict(
        model=model,
        messages=[
            dict(role='system', content=PROMPT.replace('<TARGET>', target)),
            dict(role='user', content=json.dumps(text_list, ensure_ascii=False)),
        ]
    )
    if MODELS_USE_JSON_MODE:
        kwargs['response_format'] = dict(type='json_object')
    comp = await client.chat.completions.create(**kwargs)
    logging.debug(comp)
    resp = json.loads(comp.choices[0].message.content)
    detected_lang = resp['detectedLanguage']['language']
    logging.info(
        f'{model} {detected_lang}/{target_code} '
        f'{comp.usage.prompt_tokens}+{comp.usage.completion_tokens} tokens'
    )
    if isinstance(text, str):
        resp['translatedText'] = resp['translatedText'][0]
    return resp


@routes.post('/translate')
async def translate(request: web.Request) -> web.Response:
    client = request.app[openai_app_key]
    req = await request.json()
    text, target_code = req['q'], req['target']
    if isinstance(text, str):
        text = [text]
    for model in MODELS:
        try:
            resp = await chat(client, text, target_code, model)
            return web.json_response(resp)
        except openai.RateLimitError as err:
            logging.warning(f"OpenAI rate limit: {err}")
            raise web.HTTPTooManyRequests(text="Upstream rate limit")
        except openai.OpenAIError as err:
            logging.warning(f"OpenAI error: {err}")
            raise web.HTTPServiceUnavailable(text="Upstream API error")
        except IOError as err:
            logging.warning(f"OpenAI API I/O error: {err}")
            raise web.HTTPServiceUnavailable(text="Upstream I/O error")
        except (json.JSONDecodeError, KeyError) as err:
            logging.info(f"Decoding error: {err} ({model})")
    logging.warn("All models failed")
    raise web.HTTPServiceUnavailable()


async def on_cleanup(app: web.Application):
    del app[openai_app_key]


async def init(openai_api_key: str | None) -> web.Application:
    app = web.Application()
    app.add_routes(routes)
    app[openai_app_key] = AsyncOpenAI(api_key=openai_api_key)
    app.on_cleanup.append(on_cleanup)
    return app


def main():
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

    sock = None
    if str(os.getpid()) == os.environ.get('LISTEN_PID'):
        logging.info('Systemd awared')
        fds = int(os.environ.get('LISTEN_FDS', 0))
        if fds:
            sock = socket.socket(fileno=3)
            logging.info('Use systemd-passed socket')

    openai_api_key = None
    keyfile = os.environ.get('CREDENTIALS_DIRECTORY', '') / Path('openai_key')
    if keyfile.exists():
        with keyfile.open() as f:
            openai_api_key = f.read().strip()
            logging.info(f'Load OpenAI API key from {keyfile}')

    web.run_app(init(openai_api_key), sock=sock)


if __name__ == '__main__':
    main()

