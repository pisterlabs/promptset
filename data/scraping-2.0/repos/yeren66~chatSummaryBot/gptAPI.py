import os
import openai
from dotenv import load_dotenv
 
# 加载.env文件中的环境变量
load_dotenv()
 
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")

summary = "请帮我总结以下的一段群聊对话:"
question = "请帮我解决如下问题："

# 读取文件并返回gpt总结或解决的结果
def gptWithParam(path, model="gpt-3.5-turbo", mode=summary) -> str:
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()

    completion = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": summary + "\n" + content}]
        )
        
    return completion.choices[0].message.content

# 常规使用gpt
def gpt(content, model='gpt-3.5-turbo') -> str:
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": content}]
        )
        
    return completion.choices[0].message.content