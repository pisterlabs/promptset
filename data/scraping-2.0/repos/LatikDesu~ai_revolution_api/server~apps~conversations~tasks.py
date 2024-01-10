import json

import httpx
from openai import OpenAI

from src.settings import APIKEY, PROXY_URL

proxies = {
    "all://": PROXY_URL,
}

client = OpenAI(
    api_key=APIKEY,
    http_client=httpx.Client(proxies=proxies),
)


async def send_gpt_request_async(message_list, config):
    response = client.chat.completions.create(
        model=config["model"],
        temperature=config["temperature"],
        max_tokens=config["maxTokens"],
        top_p=config["topP"],
        frequency_penalty=config["frequencyPenalty"],
        presence_penalty=config["presencePenalty"],
        stream=True,
        messages=[
            {
                "role": "system",
                "content": f"{config['prompt']}",
                "role": "user",
                "content": "The response should be returned in markdown formatting.",
            },
        ]
        + message_list,
    )
    for chunk in response:
        chunk_parsed = {
            "role": "assistant",
            "content": chunk.choices[0].delta.content,
            "finish_reason": chunk.choices[0].finish_reason,
        }
        yield f"data: {json.dumps(chunk_parsed)}\n\n"
