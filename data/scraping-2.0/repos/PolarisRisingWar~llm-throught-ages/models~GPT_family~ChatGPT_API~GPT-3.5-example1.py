import openai,tiktoken,json

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from key import *

openai.api_key=API_token
openai.proxy=API_proxy

encoding=tiktoken.get_encoding("cl100k_base")

role_content="写小说手速太慢，一小时只能写500字，如何提升速度？"  #示例输入文本

@retry(wait=wait_random_exponential(min=60,max=6000),stop=stop_after_attempt(6))
def avoid_proxy_die(model_name:str,role_content:str):
    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role":"user","content":role_content}
        ]
    )
    return completion

    
if len(encoding.encode(role_content))>4097-7:  #这个7是ChatGPT API会自带的一些内容占的长度，具体数字是靠运行一遍API算出来的
    model_name="gpt-3.5-turbo-16k"
else:
    model_name="gpt-3.5-turbo"


completion=avoid_proxy_die(model_name,role_content)
print(completion.choices[0].message['content'])  #ChatGPT返回文本
    
    