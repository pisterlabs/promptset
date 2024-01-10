from openai import OpenAI

# If runnning this service with proxy, you might need to unset `http(s)_proxy`.
base_url = "http://localhost:22222"
api_key = "sk-xxxxx"

client = OpenAI(base_url=base_url, api_key=api_key)
response = client.chat.completions.create(
    model="precise",
    messages=[
        {
            "role": "user",
            "content": "search california's weather for me",
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
