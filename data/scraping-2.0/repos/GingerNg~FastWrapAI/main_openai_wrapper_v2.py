import sys
import os

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from sse_starlette.sse import EventSourceResponse
from utils.env_utils import EnvKeys, EnvContext, app
from utils.logger_utils import get_logger
from utils.openai_utils import num_tokens_from_messages
logger = get_logger()

import json
from typing import List, Optional, Union

from dataclasses import dataclass
from typing import List, Dict
from enum import Enum
from controllers import user_controller
from common.protocol.openai_api_protocol import ChatCompletionStreamResponse, CompletionStreamResponse
from common.protocol.openai_api_protocol import OpenaiGeneratorPath
# *******************************************

@dataclass
class Context:
    llm_model_type: str
    model: any
    tokenizer: any
    embeddings_model: any
    tokens: List[str]

from common.domains import UserTokens

# *******************************************

context = Context(None, [], None, None, UserTokens.keys())

def generate_response(message: Union[str,Dict]=None, path=OpenaiGeneratorPath.TEXT_CHAT_COMPLETION.value,
                    model = "gpt-3.5-turbo-0301",
                    object= "chat.completion"):
    pass


def generate_stream_response_start():
    return {
        "id": "chatcmpl-77QWpn5cxFi9sVMw56DZReDiGKmcB",
        "object": "chat.completion.chunk", "created": 1682004627,
        "model": "gpt-3.5-turbo-0301",
        "choices": [{"delta": {"role": "assistant"}, "index": 0, "finish_reason": None}]
    }


def generate_stream_response(content: str):
    return {
        "id": "chatcmpl-77QWpn5cxFi9sVMw56DZReDiGKmcB",
        "object": "chat.completion.chunk",
        "created": 1682004627,
        "model": "gpt-3.5-turbo-0301",
        "choices": [{"delta": {"content": content}, "index": 0, "finish_reason": None}
                    ]}


def generate_stream_response_stop():
    return {"id": "chatcmpl-77QWpn5cxFi9sVMw56DZReDiGKmcB",
            "object": "chat.completion.chunk", "created": 1682004627,
            "model": "gpt-3.5-turbo-0301",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]
            }

class Message(BaseModel):
    role: str
    content: str



from common.domains import User, db, db_app
from common.dao import FsaDao
fsa_dao = FsaDao(db, db_app)
from typing import Dict, Any
import httpx
headers = {"User-Agent": "Portal API Server"}

from common.protocol.worker_api_protocol import SvcModelMapping, ModelSvcMapping
svc2model_mapping = SvcModelMapping(json.load(open("conf/svc2model.json", "r")))
model2svc_mapping = svc2model_mapping.reverse()
logger.debug(f"model2svc_mapping: {model2svc_mapping}")

from common.factory import get_svc_rd
svc_rd = get_svc_rd()

WORKER_API_TIMEOUT = 30

# ******************************************* generator ****
from common.protocol.worker_api_protocol import WorkerGeneratorPath
from common.domains import Dialog
def save_dialog_count(msg, payload, user: User):
    """count tokens
    save content to db"""
    # messages = payload["messages"]
    # messages.append(msg)
    # token_used = num_tokens_from_messages(messages=messages)
    token_used = -1
    dialog = Dialog(payload=payload, message=msg, user_id=user.id, model=payload["model"], token_used=token_used)
    fsa_dao.save_obj(dialog)
    logger.debug("saved dialog to db")

async def generate_completion_stream(payload: Dict[str, Any], worker_addr, user: User):
    # worker_addr = svc_rd.get_svc(model2svc_mapping.get(payload["model"]))
    try:
        tokens = []
        async with httpx.AsyncClient() as client:
            # worker_addr = await _get_worker_address(payload["model"], client)
            delimiter = b"\0"
            async with client.stream(
                "POST",
                worker_addr + WorkerGeneratorPath.TEXT_COMPLETION_STREAM.value,
                headers=headers,
                json=payload,
                timeout=WORKER_API_TIMEOUT,
            ) as response:
                # content = await response.aread()
                if response.status_code == 200:
                    async for raw_chunk in response.aiter_raw():
                        for chunk in raw_chunk.split(delimiter):
                            if not chunk:
                                continue
                            data = chunk.decode()
                            logger.debug(data)
                            resp = json.loads(data)  # StreamCompletionRet
                            path = payload.get("path", None)
                            if path == OpenaiGeneratorPath.TEXT_COMPLETION.value:
                                text = resp["data"][0]["text"]
                                if text:
                                    tokens.append(text)
                                # StreamCompletionRet --> CompletionStreamResponse
                                yield json.dumps(CompletionStreamResponse(choices=resp["data"],
                                                                    model=payload["model"])
                                                                    .dict(exclude_unset=True))
                            elif path is None or path == OpenaiGeneratorPath.TEXT_CHAT_COMPLETION.value:
                                delta = resp["data"][0]["delta"]
                                if "content" in delta:
                                    tokens.append(delta["content"])
                                # StreamCompletionRet --> ChatCompletionStreamResponse
                                yield json.dumps(ChatCompletionStreamResponse(choices=resp["data"],
                                                                model=payload["model"],
                                                                ).dict(exclude_unset=True))
                else:
                    raise HTTPException(response.status_code, response.text)
        # msg = {"content": "".join(tokens)}
        # save_dialog_count(msg, payload, user) todo
    except httpx.TimeoutException:
        raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, "Time out!")
    except Exception as e:
        # print(e)
        logger.error(e.__repr__())
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"ERROR! detail: {e.__repr__()}")
    # print("-----------------------")
    # yield json.dumps(generate_stream_response(e.__repr__()), ensure_ascii=False)
    # yield json.dumps(generate_stream_response_stop(), ensure_ascii=False)


async def generate_completion(payload: Dict[str, Any], worker_addr, user: User) -> str:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                worker_addr + WorkerGeneratorPath.TEXT_COMPLETION.value,
                headers=headers,
                json=payload,
                timeout=WORKER_API_TIMEOUT,
            )
            if response.status_code == 200:
                completion = response.json()
                logger.debug(completion) # CompletionRet
                # save_dialog_count(completion, payload, user)
                return completion
            else:
                raise HTTPException(response.status_code, response.text)
    except httpx.TimeoutException:
        raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, "Time out!")
    except Exception as e:
        logger.error(e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"ERROR! detail: {e.__repr__()}")


async def generate_embedding(payload: Dict[str, Any], worker_addr, user: User):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                worker_addr + WorkerGeneratorPath.TEXT_EMBEDDING.value,
                headers=headers,
                json=payload,
                timeout=WORKER_API_TIMEOUT,
            )
            if response.status_code == 200:
                resp = response.json()
                logger.debug(resp) # EmbeddingRet: resp = {"data": {"embeddings": [[0.1, 0.2, 0.3]]}, ..., "code": 1}
                # save_dialog_count(completion, payload, user)
                return resp
            else:
                raise HTTPException(response.status_code, response.text)
    except httpx.TimeoutException:
        raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, "Time out!")
    except Exception as e:
        # print(e)
        logger.error(e.__repr__())
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"ERROR! detail: {e.__repr__()}")
# ************************************** endpoints **************************************
from common.protocol.openai_api_protocol import UsageInfo
from common.protocol.openai_api_protocol import EmbeddingsRequest, ChatCompletionRequest, CompletionRequest
from common.protocol.openai_api_protocol import CompletionResponse, EmbeddingsResponse, ChatCompletionResponse
class ChatBody(BaseModel):
    messages: List[Message]
    model: str
    stream: Optional[bool] = False
    max_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]
    functions: Optional[List[Dict]]
    function_call: Optional[str]


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request, background_tasks: BackgroundTasks):
    """
    https://platform.openai.com/docs/api-reference/chat
    """
    auth_token = request.headers.get("Authorization").split(" ")[1]
    # if auth_token not in context.tokens:
    user = fsa_dao.query_by_key(token=auth_token)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token is wrong!")

    # find user and limit by token
    if user.token_limit - user.token_used < 0:
        raise HTTPException(status.HTTP_402_PAYMENT_REQUIRED, "cash used up!")

    # todo: check model
    model = body.model
    # if model not in context.model:
    #     raise HTTPException(status.HTTP_404_NOT_FOUND, "LLM model not found!")

    # question = body.messages[-1]
    # if question.role == 'user':
    #     question = question.content
    # else:
    #     raise HTTPException(status.HTTP_400_BAD_REQUEST, "No Question Found")
    # messages = []
    # for message in body.messages:
    #     messages.append(message.__dict__)

    # history = []
    # user_question = ''
    # for message in body.messages:
    #     if message.role == 'system':
    #         history.append((message.content, "OK"))
    #     if message.role == 'user':
    #         user_question = message.content
    #     elif message.role == 'assistant':
    #         assistant_answer = message.content
    #         history.append((user_question, assistant_answer))

    # logger.debug(f"question = {question}, history = {history}")

    # print(body)
    logger.debug(body.dict())
    payload = {}
    payload.update(body.dict(exclude_defaults=True))

    worker_addr = svc_rd.get_svc(model2svc_mapping.get(payload["model"]))
    logger.debug(f"worker_addr = {worker_addr}")
    if worker_addr is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "current model is not available")
    if body.stream:
        generator = generate_completion_stream(payload=payload,
                                    worker_addr=worker_addr,
                                    user=user)
        return EventSourceResponse(generator, ping=10000)
    else:
        # response = "hello"
        response = await generate_completion(payload=payload,
                                            worker_addr=worker_addr,
                                            user=user)
        logger.debug(f"response = {response}")
        # CompletionRet --> ChatCompletionResponse
        return JSONResponse(content=ChatCompletionResponse(
            model=model,
            choices=response["data"],
            usage=UsageInfo(**response["usage_info"])
        ).dict())

class CompletionBody(BaseModel):
    prompt: Union[str, List]
    model: str
    stream: Optional[bool] = False
    max_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[int]
    n: Optional[int] = 1
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    frequency_penalty: Optional[int]
    presence_penalty :  Optional[int]
    logit_bias: Optional[Dict] = None
    stop: Optional[List]


@app.post("/v1/completions")
async def completions(body: CompletionBody, request: Request, background_tasks: BackgroundTasks):
    """
    https://platform.openai.com/docs/api-reference/completions
    """
    # receive_ = await request._receive()
    # print(receive_)
    # return {}
    # background_tasks.add_task(torch_gc)
    auth_token = request.headers.get("Authorization").split(" ")[1]
    # if auth_token not in context.tokens:
    user = fsa_dao.query_by_key(token=auth_token)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token is wrong!")

    # find user and limit by token
    # user = UserTokens.get(auth_token, None)
    if user.token_limit - user.token_used < 0:
        raise HTTPException(status.HTTP_402_PAYMENT_REQUIRED, "cash used up!")

    # todo: check model
    model = body.model
    # if model not in context.model:
    #     raise HTTPException(status.HTTP_404_NOT_FOUND, "LLM model not found!")

    payload = {"path": "/v1/completions"}
    payload.update(body.dict(exclude_defaults=True))

    worker_addr = svc_rd.get_svc(model2svc_mapping.get(payload["model"]))
    if worker_addr is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "current model is not available")
    if body.stream:
        generator = generate_completion_stream(payload=payload,
                                    worker_addr=worker_addr,
                                    user=user)
        return EventSourceResponse(generator, ping=10000)
    else:
        response = await generate_completion(payload=payload,
                                            worker_addr=worker_addr,
                                            user=user)
        logger.debug(f"response = {response}, {type(response)}")
        # CompletionRet --> CompletionResponse
        return JSONResponse(content=CompletionResponse(
            model=model,
            choices=response["data"], #  dict
            usage=UsageInfo(**response["usage_info"])
        ).dict())


@app.post("/v1/embeddings")
async def completions(body: EmbeddingsRequest, request: Request, background_tasks: BackgroundTasks):
    """
    https://platform.openai.com/docs/api-reference/embeddings
    """
    auth_token = request.headers.get("Authorization").split(" ")[1]
    # if auth_token not in context.tokens:
    user = fsa_dao.query_by_key(token=auth_token)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token is wrong!")

    # find user and limit by token
    # user = UserTokens.get(auth_token, None)
    if user.token_limit - user.token_used < 0:
        raise HTTPException(status.HTTP_402_PAYMENT_REQUIRED, "cash used up!")

    # todo: check model
    model = body.model
    # if model not in context.model:
    #     raise HTTPException(status.HTTP_404_NOT_FOUND, "LLM model not found!")

    payload = {"path": "/v1/embeddings"}
    payload.update(body.dict(exclude_defaults=True))

    worker_addr = svc_rd.get_svc(model2svc_mapping.get(payload["model"]))
    if worker_addr is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "current model is not available")

    response = await generate_embedding(payload=payload,
                                        worker_addr=worker_addr,
                                        user=user)
    logger.debug(f"response = {response}, {type(response)}")
    return JSONResponse(content=EmbeddingsResponse(
        model=model,
        data=response["data"],
        usage=UsageInfo(**response["usage_info"])
    ).dict())



# *******************************************************************************

import sys
def usage():
    """
    print usage message and exit.
    """
    print('Usage: {} svc-host svc-port svc-name(optional)'.format(sys.argv[0]))
    sys.exit(1)
if __name__ == '__main__':
    if len(sys.argv) < 3:
        usage()
    svc_port = int(sys.argv[2])
    host=sys.argv[1]
    if len(sys.argv) == 4:
        name = sys.argv[3]
    else:
        name = "OpenAI-compatible-API-server"
    svc_rd.register_svc(name, f"{name}-{host}-{svc_port}", host, svc_port)
    uvicorn.run(app, host="0.0.0.0", port=svc_port)

