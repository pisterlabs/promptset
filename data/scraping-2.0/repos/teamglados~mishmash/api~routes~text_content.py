from api.app import *

import openai
import asyncio
import concurrent
import functools
import os


def strip_non_alnum_from_text(text):
    # avoid outputs like: ':\n\n\n\nNew iPhone: Now with 100% more battery life!'
    first_alnum_character = 0
    for idx, s in enumerate(text):
        if s.isalnum():
            first_alnum_character = idx
            break
    return text[first_alnum_character:].strip()


@functools.lru_cache
def generate_text_content(api_key, **kwargs):
    """
    Inlcude previous prompt input separeted by \n\n

    Examples:
    "Create funny one sentence marketing text for new iPhone"

    "Create marketing text for new iPhone model\n\nThe new iPhone model is the most powerful and \
    sophisticated iPhone yet. It has a powerful A12 processor and a new design that is sleek and \
    stylish. This phone is sure to revolutionize the smartphone industry.\n\ntranslate to spanish"
    """
    try:
        openai.api_key = api_key
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=kwargs["prompt"],
            max_tokens=256,
            temperature=0.7,
            top_p=1,
            n=1,
            stream=False,
            logprobs=None,
            echo=False,
        )
        res_dict = response.to_dict()
        logger.info(f"Completion API completed with: {res_dict}")
        if res_dict["choices"]:
            res_text = res_dict["choices"][0].to_dict()["text"].strip()
            return strip_non_alnum_from_text(res_text)
        return None
    except:
        logger.exception("Completion api failed:")
        return None


async def create_text(*args, **kwargs):
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as exc:
        return await asyncio.get_running_loop().run_in_executor(
            exc, functools.partial(generate_text_content, os.environ["OPENAI_KEY"], **kwargs)
        )


@app.get("/textcontent")
async def create(prompt: str = ""):
    logger.info(f"creating text using the following prompt: {repr(prompt)}")
    result = await create_text(prompt=prompt)
    return {'result': result}
