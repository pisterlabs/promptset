import json
import os

from typing import Any
from pydantic import BaseModel
from time import time
from fastapi import Query, APIRouter, Body, Response
from fastapi.responses import StreamingResponse
from starlette import status

from fastapi_app.utils.logger import setup_logging
from fastapi_app.responses.api_responses import CHAT_RESPONSES, CHAT_RESPONSES_SIMPLE
from fastapi_app.chatbot.assistant import get_answer_simple
from fastapi_app.chatbot.custom_langchain import answer_with_openai, answer_with_openai_translated, \
    format_answer_with_openai
from fastapi_app.chatbot.second_chance import second_chance
from fastapi_app.chatbot.update_sources import enrich_sources as enrich_sources_func
from fastapi_app.chatbot.fake_keys.validate_key import use_key

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "OPENAI_API_KEY")
router = APIRouter()
logger = setup_logging()


class PrettyJSONResponse(Response):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=4,
            separators=(", ", ": "),
        ).encode("utf-8")


class QuestionParams(BaseModel):
    tada_key: str = Query('54321test', description="Tada key")
    topic: str = Query('business', example='tk', description="Choose topic: [business, tk, hr, yt]")
    enrich_sources: bool = Query(True, description="Add links to sources (tk, yt only)")


class QuestionParamsSimple(BaseModel):
    tada_key: str = Query(..., description="API key")


class DebugParams(QuestionParams):
    api_key: str = Query(OPENAI_API_KEY, description="API key")
    html: bool = Query(False, description="Generate html")
    verbose: bool = Query(False, description="Verbose")
    temperature: float = Query(0.01, description="Temperature")
    return_context: bool = Query(False, description="Return context")


def _get_valid_key(key):
    key_status = use_key(key)
    if 'success' in key_status:
        logger.info(f"key: {key_status['success']}")
        return OPENAI_API_KEY, key_status['uses_left']
    else:
        logger.info(f"key: {key_status['error']}")
        return None, key_status['error']


def get_answer_with_sources(user_input, api_key, topic, translate_answer=False):
    if topic == 'yt':
        answer, sources = answer_with_openai_translated(question=user_input, api_key=api_key, faiss_index=topic,
                                                        translate_answer=translate_answer)
        answer = format_answer_with_openai(answer, api_key=api_key) if not translate_answer else answer
    else:
        answer, sources = answer_with_openai(question=user_input, api_key=api_key, faiss_index=topic)
    answer, sources = second_chance(answer, sources, user_input, api_key)
    return answer, sources


@router.post('/chatbot_simple/{user_id}', include_in_schema=True, responses=CHAT_RESPONSES_SIMPLE)
async def ask_chatbot(
        user_id: str,
        user_input: str = Body('как начисляется ндфл сотруднику работающему из другой страны',
                               example="How are you?",
                               description="User text input",
                               max_length=500),
        params: QuestionParamsSimple = Body(...),
):
    """
    API endpoint for AI assistant (simple version)
    """
    api_key, uses_left = _get_valid_key(params.tada_key)
    if api_key is None:
        return PrettyJSONResponse(content={"error": "Your key is invalid or expired",
                                           "key_status": uses_left})

    config = {"user_id": user_id,
              "user_input": user_input,
              "user_key": params.tada_key,
              }
    print("user request:", config)
    logger.info(f"user request: {config}")

    start_time = time()
    result = get_answer_simple(question=user_input, api_key=api_key)
    elapsed_time = time() - start_time

    try:
        result.update({"user_request": config, "key_status": uses_left, "elapsed_time": elapsed_time})
    except Exception as e:
        print(f"Error decoding: {e}")
        logger.error(f"Error decoding: {e}")

    logger.info(f"response: {result}")
    return PrettyJSONResponse(content=result)


@router.post('/chatbot_topic/{user_id}', include_in_schema=True, responses=CHAT_RESPONSES)
async def ask_assistant(
        user_id: str,
        user_input: str = Body('как начисляется ндфл сотруднику работающему из другой страны',
                               example="How are you?",
                               description="User text input",
                               max_length=1500),
        params: QuestionParams = Body(...),
):
    """
    API endpoint for AI assistant (advanced version)
    """
    api_key, uses_left = _get_valid_key(params.tada_key)
    if api_key is None:
        return PrettyJSONResponse(content={"error": "Your key is invalid or expired",
                                           "key_status": uses_left})
    config = {
        "user_input": user_input,
        "topic": params.topic,
        "user_id": user_id,
        "user_key": params.tada_key,
    }
    print("user request:", config)
    logger.info(f"user request: {config}")

    start_time = time()
    answer, sources = get_answer_with_sources(user_input=user_input, api_key=api_key, topic=params.topic)
    elapsed_time = time() - start_time
    sources = enrich_sources_func(sources, params.topic)

    response_content = {"answer": answer, "sources": sources, "user_request": config, "uses_left": uses_left,
                        "elapsed_time": elapsed_time}

    logger.info(f"response: {response_content}")
    return PrettyJSONResponse(content=response_content)


async def calling_assistant(user_input: str, topic: str = "default", enrich_sources: bool = True,
                            tada_key: str = "ratelimit"):
    api_key, uses_left = _get_valid_key(tada_key)
    if api_key is None:
        logger.warning("Превышен лимит запросов, попробуйте позже")
        raise PermissionError("Превышен лимит запросов, попробуйте позже")

    answer, sources = answer_with_openai(question=user_input, api_key=api_key, faiss_index=topic)
    answer, sources = second_chance(answer, sources, user_input, api_key)
    sources = enrich_sources_func(sources, topic) if enrich_sources else sources

    response_content = {"answer": answer, "sources": sources}
    logger.info(f"response: {response_content}, {uses_left=}")

    return response_content


@router.post('/chatbot_stream/{user_id}', include_in_schema=False, responses=CHAT_RESPONSES)
async def streaming_assistant(
        user_input: str = Query('как начисляется ндфл сотруднику работающему из другой страны',
                                example="How are you?",
                                description="User text input",
                                max_length=1500),
):
    import openai

    async def generate():
        # Looping over the response
        context = "You an expert in the field of business, finance, law, and HR. " \
                  "You are answering questions from a client."
        task = "Answer the question. Be specific and use bullet points, return answer in Russian\n\n"
        info = f"Context: {context}\n\n---\n\nQuestion: {user_input}\nAnswer:"
        messages = [{"role": "system", "content": f"{task}{info}"}]
        openai.api_key = os.getenv("OPENAI_API_KEY")
        for resp in openai.ChatCompletion.create(model='gpt-3.5-turbo',
                                                 messages=messages,
                                                 max_tokens=1200,
                                                 temperature=0.01,
                                                 stream=True):
            # Assuming here you're yielding the text returned by the model
            # The actual data structure of `resp` would need to be examined
            yield resp

    return StreamingResponse(generate(), media_type="application/json")
