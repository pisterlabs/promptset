# A collection of inference functions to be called from the main Streamlit app

import os
import json
import aiohttp
import importlib
import streamlit as st
from settings import *

def convert_openai_messages_to_prompt(messages: list) -> str:
    # Convert the OpenAI standard messages format into a text prompt for models which don't support it
    prompt = "Below is a conversation between a human (user) and an AI (assistant). Please continue the conversation by ONE REPLY only. Do not add any follow-up comments or other extra contents.\n"
    for message in messages:
        match message["role"]:
            case "user":
                prompt += f"User: {message['content']}\n"
            case "assistant":
                prompt += f"Assistant: {message['content']}\n"
            # For now, we ignore other possible roles, but may revisit the logic later
    prompt += "Assistant: "
    return prompt


async def call_openai(
    messages: list,
    model_settings: dict,
) -> str | None:
    try:
        # Just make sure the environment variables are set, OpenAI API can get them automatically
        os.getenv("OPENAI_API_KEY")
    except KeyError as e:
        raise RuntimeError(f"Missing environment variable: {e}")

    import openai
    importlib.reload(openai)
    if "openai_key" in st.session_state:
        openai.api_key = st.session_state["openai_key"]
        openai.api_base = st.session_state["OPENAI_API_BASE"].rstrip("/")
        openai.api_type = st.session_state["OPENAI_API_TYPE"]
        openai.api_version = None
    
    stream = model_settings.get("stream", False)
    engine = model_settings.get("model_engine", None)
    if engine is None:
        call_func = openai.ChatCompletion.acreate(
            model=model_settings["model_name"],
            messages=messages,
            max_tokens=model_settings.get("max_reply_tokens", LLM_MAX_REPLY_TOKENS),
            stream=stream,
            timeout=TIMEOUT,
        )
    else:
        call_func = openai.ChatCompletion.acreate(
            model=model_settings["model_name"],
            engine=model_settings["model_engine"],
            messages=messages,
            max_tokens=model_settings.get("max_reply_tokens", LLM_MAX_REPLY_TOKENS),
            stream=stream,
            timeout=TIMEOUT,
        )
    if stream:
        async for chunk in await call_func:
            content = chunk["choices"][0].get("delta", {}).get("content", None)
            if content is not None:
                yield content
    else:
        resp = await call_func
        yield resp["choices"][0]["message"]["content"].strip()


async def call_deepinfra(
    messages: list,
    model_settings: dict,
):
    try:
        api_key = os.getenv("DEEPINFRA_API_KEY")
        api_base = os.getenv("DEEPINFRA_API_BASE").rstrip("/")
    except KeyError as e:
        raise RuntimeError(f"Missing environment variable: {e}")
    
    method = "POST"
    url=f"{api_base}/{model_settings['model_name']}"
    stream = model_settings.get("stream", False)
    
    headers = {
        "Authorization": f"bearer {api_key}",
        "Content-Type": "application/json",
    }

    prompt = convert_openai_messages_to_prompt(messages)

    data = {
        "input": prompt,
        "max_new_tokens": model_settings.get("max_reply_tokens", LLM_MAX_REPLY_TOKENS),
        "stream": stream,
    }
    if "stop" in model_settings:
        data["stop"] = model_settings["stop"]

    async for resp in _call_api(
        method=method,
        url=url,
        headers=headers,
        data=data,
        stream=stream,
    ):
        if stream:
            if isinstance(resp, str) and resp.startswith("data: "):
                resp = json.loads(resp.split("data: ", 1)[1])
                content = resp.get("token", {}).get("text", None)
                if content is not None:
                    if content == "</s>":   # Special eos token
                        break
                    if content.endswith("</s>"):
                        content = content.split("</s>", 1)[0]
                    yield content
        else:
            # Convert dictionary-typed responses into strings
            if isinstance(resp, dict):
                if "results" in resp:
                    resp = resp["results"][0]["generated_text"].strip()
                elif "generated_text" in resp:
                    resp = resp["generated_text"].strip()
            # Sometimes the model might return the entire prompt together with the response, we need to parse it out
            if isinstance(resp, str) and resp.startswith(prompt):
                resp = resp.split(prompt, 1)[1]
            yield resp


async def _call_api(
    method: str,
    url: str,
    headers: dict = {},
    params: dict = {},
    data: dict = {},
    stream: bool = False,
):
    async with aiohttp.ClientSession() as session:
        async with session.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=data,
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"API call failed with status {resp.status}: {await resp.text()}")
            if stream:
                async for line in resp.content:
                    chunk = line.decode("utf-8").strip()
                    yield chunk
            else:
                yield await resp.json()