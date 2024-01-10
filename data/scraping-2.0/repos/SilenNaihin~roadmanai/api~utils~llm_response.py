import os
import openai
from typing import List, Dict

os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def model_response(
    chat: List[Dict[str, str]],
    stream: bool = False,
    model: str = "gpt-3.5-turbo",
    max_tokens: int = 200,
    temperature: float = 0.5,
) -> str:
    try:
        kwargs = {
            # settings
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": chat,
            "api_key": OPENAI_API_KEY,
            "stream": stream,
        }

        result = openai.ChatCompletion.create(**kwargs)

        if result is None:
            print("ERROR: The function call to the completion API timed out")
            return ""

        return result.choices[0].message.content  # type: ignore
    except Exception as e:
        return f"{e}"
