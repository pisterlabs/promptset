import codecs

import openai
from pydantic import BaseModel, parse_obj_as


class ApiConfig(BaseModel):
    openai_organization_id: str
    openai_key: str


class LlmMessage(BaseModel):
    role: str
    content: str


def get_completion(
    # TODO: Figure out how to apply APK keys per request, instead of globally.
    api_config: ApiConfig,
    model: str,
    temperature: float,
    messages: list[LlmMessage],
    stop: str,
) -> LlmMessage:
    response: dict = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        messages=[m.dict() for m in messages],
        # NOTE: Must unescape the stop token, otherwise OpenAI API won't match
        # tokens like \n, because the stop str is stored as \\n in the database.
        stop=codecs.decode(stop, "unicode_escape") if stop != "" else "",
    )

    choice = response["choices"][0]
    return parse_obj_as(LlmMessage, choice["message"])
