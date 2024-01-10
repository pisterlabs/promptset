import os
from typing import Annotated, List, Union

import openai
from dotenv import load_dotenv
from fastapi import Body, FastAPI
from pydantic import BaseModel


class ModerationEntity(BaseModel):
    input: str


async def check_abuse_api(text: ModerationEntity):
    response = openai.Moderation.create(
        model="text-moderation-latest",
        input=text.input,
    )

    is_flagged = response.results[0].flagged

    return {"is_flagged":is_flagged}


