import logging
from functools import wraps

import openai
from django.conf import settings

from ..models import OpenAICall

logger = logging.getLogger('axiom')

openai.api_key = settings.OPENAI_KEY


def log_to_db(func):
    @wraps(func)
    async def _f(user_id: int, *args, **kwargs):
        err = None
        try:
            response = await func(*args, **kwargs)
        except openai.OpenAIError as e:
            err = e
            logger.error('openai error %s', ' '.join(e.args))
            response = {
                'error': str(e),
            }

        await OpenAICall.objects.acreate(
            user_id=user_id,
            request={
                'args': args,
                'kwargs': kwargs,
            },
            response=response,
            tokens=response.get('usage', {}).get('total_tokens', 0),
        )

        if err:
            raise err
        return response

    return _f


@log_to_db
async def create_chat_completion(model: str, messages: list[dict[str, str]]):
    return await openai.ChatCompletion.acreate(
        model=model,
        messages=messages,
    )


@log_to_db
async def create_image(prompt: str, n: int, size: str):
    return await openai.Image.acreate(
        prompt=prompt,
        n=n,
        size=size,
    )


@log_to_db
async def create_image_variation(image: bytes, n: int, size: str):
    return await openai.Image.acreate_variation(
        image=image,
        n=n,
        size=size,
    )
