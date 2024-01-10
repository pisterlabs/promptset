import json
import google.generativeai as palm
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from config import SECRET_TOKEN, OPEN_API_KEY, FIREWORKS_API_KEY, PALM_KEY
from openai import AsyncOpenAI

from typing import Iterator

palm.configure(api_key=PALM_KEY)

async def create(client ,model, body: json):
    create_kwargs = {
        "model": model,
        "messages": body["messages"],
    }
    if "frequency_penalty" in body:
        create_kwargs["frequency_penalty"] = body["frequency_penalty"]
    if "logit_bias" in body:
        create_kwargs["logit_bias"] = body["logit_bias"]
    if "max_tokens" in body:
        create_kwargs["max_tokens"] = body["max_tokens"]
    if "n" in body:
        create_kwargs["n"] = body["n"]
    if "presence_penalty" in body:
        create_kwargs["presence_penalty"] = body["presence_penalty"]
    if "response_format" in body:
        create_kwargs["response_format"] = body["response_format"]
    if "seed" in body:
        create_kwargs["seed"] = body["seed"]
    if "stop" in body:
        create_kwargs["stop"] = body["stop"]
    if "stream" in body:
        create_kwargs["stream"] = body["stream"]
    if "temperature" in body:
        create_kwargs["temperature"] = body["temperature"]
    if "top_p" in body:
        create_kwargs["top_p"] = body["top_p"]
    if "tools" in body:
        create_kwargs["tools"] = body["tools"]
    if "tool_choice" in body:
        create_kwargs["tool_choice"] = body["tool_choice"]
    if "user" in body:
        create_kwargs["user"] = body["user"]
    ## Deprecated
    if "function_call" in body:
        create_kwargs["function_call"] = body["function_call"]
    ## Deprecated
    if "functions" in body:
        create_kwargs["functions"] = body["functions"]
    return await client.chat.completions.create(**create_kwargs)

async def createGoogle(model, body: json):
    requestMessages = body["messages"]
    messages = []
    context = ""
    for message in requestMessages:
        auther = ""
        if message["role"] == 'system':
            context = message["content"]
            continue
        elif message["role"] == 'user':
            auther = "1"
        elif message["role"] == 'assistant':
            auther = "2"
        messages.append({'author': auther, 'content': message["content"]})
    palmResponce = palm.chat(model=model,
                             messages=messages,
                             context=context
                            #  temperatur=body["temperature"],
                            #  top_p=body["top_p"],
                             )
    print(palmResponce.messages)
    return {
        "id": "chatcmpl-9999",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "system_fingerprint": "fp_9999999",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": palmResponce.last,
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
         }
}

async def chat_stream_generator(client, model, body: json) -> Iterator[bytes]:
   response = await create(client, model, body)
   async for chunk in response:
       yield json.dumps(chunk.dict(), ensure_ascii=False).encode('utf-8')

async def get_chat_completions(request: Request):
    signature = request.headers['Authorization']
    if signature != "Bearer " + SECRET_TOKEN:
        raise HTTPException(status_code=403, detail="Not authorized")
    body_raw = await request.body()
    body = json.loads(body_raw.decode("utf-8"))
    model = body["model"]
    if model.startswith("fireworks/"):
      model = model.replace("fireworks/", "accounts/fireworks/models")
      client = AsyncOpenAI(base_url = "https://api.fireworks.ai/inference/v1/", api_key = FIREWORKS_API_KEY)
    elif model.startswith("openai/"):
      model = model.replace("openai/", "")
      client = AsyncOpenAI(base_url = "https://api.openai.com/v1/", api_key = OPEN_API_KEY)
      client.chat.completions.create
    elif model.startswith("google/"):
      model = model.replace("google/", "")
      return await createGoogle(model, body)
    if body.get("stream", False):
        return StreamingResponse(chat_stream_generator(client, model, body), media_type="text/event-stream")
    return await create(client, model, body)
