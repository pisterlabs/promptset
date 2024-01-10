from openai import OpenAI

# If runnning this service with proxy, you might need to unset `http(s)_proxy`.
base_url = "http://127.0.0.1:23333"
api_key = "sk-xxxxx"

client = OpenAI(base_url=base_url, api_key=api_key)
response = client.chat.completions.create(
    model="mixtral-8x7b",
    messages=[
        {
            "role": "user",
            "content": "what is your model",
        }
    ],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
    elif chunk.choices[0].finish_reason == "stop":
        print()
    else:
        pass
