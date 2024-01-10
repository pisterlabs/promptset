from openai import OpenAI
from httpx import Client

messages = [
    {
        "content": "",
        "role": "assistant",
        "function_call": None,
        "tool_calls": [
            {
                "id": "call_qNz6JXfuqDs3z2RgLPW8Vbhe",
                "function": {"arguments": '{"input":"你好"}', "name": "tts"},
                "type": "tool",
            }
        ],
    },
    {
        "tool_call_id": "call_qNz6JXfuqDs3z2RgLPW8Vbhe",
        "role": "tool",
        "name": "tts",
        "content": "success to generate audio",
    },
]

openai_client = OpenAI(
    base_url="https://one.loli.vet/v1",
    api_key="sk-",
    http_client=Client(follow_redirects=True),
)

print(openai_client.chat.completions.create(messages=messages, model="gpt-3.5-turbo-1106"))
