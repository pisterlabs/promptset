from fastapi import HTTPException
import openai
import tiktoken
import os
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY


async def get_response_openai(prompt):
    """ Get a response from OpenAI given a prompt. Token count is checked to determine which model to use."""
    try:
        tokens = len(tiktoken.encoding_for_model("gpt-3.5-turbo").encode(prompt))
        model = "gpt-3.5-turbo" if tokens < 4000 else "gpt-3.5-turbo-16k"
        prompt = prompt
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )

        async for chunk in response:
            current_content = chunk["choices"][0]["delta"].get("content", "")
            if current_content:
                yield current_content

    except Exception as e:
        print("OpenAI Response (Streaming) Error: " + str(e))
        raise HTTPException(503, "OpenAI server is busy, try again later")


async def get_response_prompted(prompt):
    """ Get a response from OpenAI given a contextual prompt. Token count is checked to determine which model to use."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = 0
    for context in prompt:
        for role, content in context.items():
            tokens += len(encoding.encode(content))

    try:
        model = "gpt-3.5-turbo" if tokens < 4000 else "gpt-3.5-turbo-16k"
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=prompt,
            stream=True
        )

        async for chunk in response:
            current_content = chunk["choices"][0]["delta"].get("content", "")
            if current_content:
                yield current_content

    except Exception as e:
        print("OpenAI Response (Streaming) Error: " + str(e))
        raise HTTPException(503, "OpenAI server is busy, try again later")
