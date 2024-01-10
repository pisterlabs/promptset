from openai import OpenAI

client = OpenAI()

prompt = """
你的任务就是根据我输入的内容生成logo图片,同时需要注意不要在图片中出现文字内容。
提升程序员的工作效率，降低用人成本，保证工作质量
"""

response = client.images.generate(
    model="dall-e-3",
    prompt=prompt,
    n=1,
    style="vivid",
    size="1024x1024"
)

print(response)
