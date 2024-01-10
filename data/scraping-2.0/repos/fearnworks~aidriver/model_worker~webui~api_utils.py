import httpx
import json
import textwrap
from typing import List 
import openai

def wrap_and_print(text, width=100):
    paragraphs = text.split('\n')
    wrapped_paragraphs = [textwrap.fill(para, width=width) for para in paragraphs]
    return '\n'.join(wrapped_paragraphs)

async def openai_generate(model, prompt):
    import openai 
    openai.api_key = "EMPTY"
    openai.api_base = "http://127.0.0.1:8100/v1"
    import asyncio
    from openai_streaming import process_response
    from typing import AsyncGenerator

    # Define content handler
    async def content_handler(content: AsyncGenerator[str, None]):
        async for token in content:
            print(token, end="")

    print(f"USING MODEL : {model['id']}")
    description = prompt
    response = openai.Completion.create(
    model=model["id"],
    prompt=f"Instruction:\n{description}",
    
    max_tokens=2000
    )

    response_t = response.choices[0].text.strip()
    return response_t 




async def generate(sampling_params, model:str, messages: List[str]):
    url = "http://localhost:8100/v1/chat/completions"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "n": sampling_params.n,
        "max_tokens": sampling_params.max_tokens,
        "stop": sampling_params.stop,
        "stream": True,
        "presence_penalty": sampling_params.presence_penalty,
        "frequency_penalty": sampling_params.frequency_penalty,
        "user": "User",
        "best_of": sampling_params.best_of,
        "top_k": sampling_params.top_k,
        "ignore_eos": sampling_params.ignore_eos,
        "use_beam_search": sampling_params.use_beam_search,
        "stop_token_ids": sampling_params.stop_token_ids,
        "skip_special_tokens": sampling_params.skip_special_tokens
    }

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                await response.aread()
                response.raise_for_status()  # Check if the request was successful
                
                async for chunk in response.aiter_lines():
                    if chunk:
                        yield json.loads(chunk)

    except httpx.HTTPError as e:
        print(f"An error occurred while making the request: {e}")
        if e.response is not None:
            print("Additional information:")
            print(f"Status code: {e.response.status_code}")
            print(f"Headers: {e.response.headers}")
            print(f"Content: {e.response.text}")