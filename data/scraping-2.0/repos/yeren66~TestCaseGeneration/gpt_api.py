import os
import openai
from dotenv import load_dotenv
 
# 加载.env文件中的环境变量
load_dotenv()
 
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")


# 常规使用gpt
def gpt(content, model='gpt-3.5-turbo') -> str:
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": content}]
        )
        
    return completion.choices[0].message.content