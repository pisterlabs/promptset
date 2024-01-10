from typing import Dict, List, Literal, Optional, Union

import openai
from aiofauna import *
from aiofauna.llm.llm import *
from pydantic import BaseModel, Field

Size = Literal["256x256", "512x512", "1024x1024"]
Format = Literal["url", "b64_json"]

llm = LLMStack()


class CreateImageResponse(BaseModel):
    created: float = Field(...)
    data: List[Dict[str, str]] = Field(...)


class CreateImageRequest(BaseModel):
    """Creates an Image using Dall-E model from OpenAI.
    must use default values unless user prompts for a different configuration,
    will be use in case the user asks for an image that is not a logo or a photo.
    """

    prompt: str = Field(...)
    n: int = Field(default=1)
    size: Size = Field(default="1024x1024")
    response_format: Format = Field(default="url")

    async def run(self):
        response = openai.Image.create(
            **self.dict(exclude_none=True, exclude={"response_format"})
        )
        assert isinstance(response, dict)
        return CreateImageResponse(**response).data[0]["url"]
