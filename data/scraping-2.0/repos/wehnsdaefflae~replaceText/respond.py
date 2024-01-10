from typing import Generator

import openai


def respond(message: str, *args: any, **kwargs) -> Generator[str, None, None]:
    # openai.api_key_path = sys.argv[1]

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. The user provides you with text and you improve it. You figure out the perfect tone for it, potentially missing content, as well as the reaction it is supposed to evoke in its recipient. You answer with a perfect version of the text that incorporate these aspects."
        },
        {
            "role": "user",
            "content": message
        }
    ]

    try:
        for chunk in openai.ChatCompletion.create(*args, messages=messages, stream=True, request_timeout=2, **kwargs):
            content = chunk["choices"][0].get("delta", dict()).get("content")
            if content is not None:
                yield content

    except openai.error.OpenAIError as e:
        yield f"Openai error: {e}"

