import base64
import os
from typing import List

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage

from ..models.configuration.base import Configuration


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


async def exec_vision(
    messages: List[dict[str, str]],
    inputs: str,
    conf: Configuration,
    is_link: bool = False
) -> List[ChatCompletionMessage]:
    ext_to_mime = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.tiff': 'image/tiff',
        '.tif': 'image/tiff',
        '.webp': 'image/webp',
    }

    if not is_link:
        image_type = os.path.splitext(inputs)[1]
        assert image_type in ext_to_mime, f'Unsupported image type: {image_type}'

        image_type = ext_to_mime[image_type]

    image_content = {
        'type': 'image_url',
        'image_url': {
            "url": f"data:image/{image_type};base64,{encode_image(inputs)}"
            if not is_link else inputs
        }
    }

    for i in range(len(messages)):
        if messages[i]['role'] == 'user':
            content = [
                {
                    'type': 'text',
                    'text': messages[i]['content']
                },
                image_content
            ]
            messages[i]['content'] = content

    client = AsyncOpenAI(
        api_key=conf.api_key,
        max_retries=conf.max_retries,
        organization=conf.organization
    )

    response: ChatCompletion = await client.chat.completions.create(
        messages=messages,
        model=conf.model or 'gpt-4-vision-preview',
        max_tokens=conf.max_tokens,
        top_p=conf.top_p,
        frequency_penalty=conf.frequency_penalty,
        presence_penalty=conf.presence_penalty,
        temperature=conf.temperature,
    )

    return [choice.message for choice in response.choices]
