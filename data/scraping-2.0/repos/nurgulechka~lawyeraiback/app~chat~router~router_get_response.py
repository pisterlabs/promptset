# from .dependencies import parse_jwt_user_data
import os
from typing import Any

import langchain
import openai
from app.utils import AppModel
from dotenv import load_dotenv
from fastapi import Depends
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from pydantic import Field

from ..adapters.jwt_service import JWTData
from ..service import Service, get_service
from . import router

load_dotenv()


class GetUserRequest(AppModel):
    request: str
    # password: str


class GetResponse(AppModel):
    response: str


@router.post("/Chat", response_model=GetResponse)
def get_response(
    input: GetUserRequest,
    svc: Service = Depends(get_service),
) -> dict[str, str]:
    resp = svc.repository.get_response(input.request)
    return GetResponse(response=resp)
