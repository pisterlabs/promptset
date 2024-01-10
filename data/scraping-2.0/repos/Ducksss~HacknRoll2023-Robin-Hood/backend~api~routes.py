import logging

import openai
from fastapi import APIRouter, Depends, Request
from nltk.tokenize import sent_tokenize

from api.config import settings
from api.depends import get_openai
from api.utils import compute_perplexity
from models.detect import DetectIn
from models.openai import CompletionIn


prefix = "/api/v1"
logging.debug(f"Route prefix: {prefix}")

router = APIRouter(
    prefix=settings.ROUTE_PREFIX,
    responses={
        404: {"error": "Not found"},
        500: {"server_error": "Internal server error"},
    },
)


@router.post("/completion")
async def completion(
    request: CompletionIn,
    openai: openai = Depends(get_openai),
):
    prompt = "Rephrase the following paragraph "
    prompt += "in context of an {context}. ".format(context=request.context)
    prompt += (
        "The goal is to {intent} the audience who is more {audience}. ".format(
            intent=request.intent, audience=request.audience
        )
    )
    prompt += "Use a more {formality} tone in the text. ".format(
        formality=request.formality
    )
    if request.role:
        prompt += "Role play as a {roleplay}".format(roleplay=request.role)
    prompt += '\ntext:"""\n{text}"""'.format(text=request.text)

    # Call OpenAI's API
    completion = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=2048,
        temperature=0.9,
    )

    return {"completion": completion.choices[0].text.replace("\n\n", "")}


@router.post("/detect")
def detect(
    request: DetectIn,
    raw_request: Request,
):
    sentences = sent_tokenize(request.text)
    result = compute_perplexity(
        session=raw_request.app.package["session"],
        tokenizer=raw_request.app.package["tokenizer"],
        predictions=sentences,
    )

    return {"result": result}