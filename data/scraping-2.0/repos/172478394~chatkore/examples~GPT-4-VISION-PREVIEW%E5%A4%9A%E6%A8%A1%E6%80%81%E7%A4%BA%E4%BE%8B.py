from openai import OpenAI

client = OpenAI(
    # 输入转发API Key
    api_key="sk-xxxx",
    base_url="https://api.chatkore.com/v1/chat/completions"
)

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {
                    "type": "image_url",
                    "image_url": "https://xxxx.jpg",
                },
            ],
        }
    ],
    max_tokens=300,
    stream=False  # # 是否开启流式输出
)
# 非流式输出获取结果
print(response.choices[0].message)
# 流式输出获取结果
# for chunk in response:
#     print(chunk.choices[0].delta)

