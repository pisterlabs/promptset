import openai
import json
import os

# 目前需要设置代理才可以访问 api
#os.environ["HTTP_PROXY"] = "代理地址"
#os.environ["HTTPS_PROXY"] = "代理地址"


def get_api_key():
    # 存在一个 openai_key 文件里，json 格式
    '''
    {"api": "api keys"}
    '''
    openai_key_file = 'openai_key.json'
    with open(openai_key_file, 'r', encoding='utf-8') as f:
        openai_key = json.loads(f.read())
    return openai_key['api']

openai.api_key = get_api_key()

#q是问题，prompt在这个地方输入
q = "1+1等于几，简短回答"
rsp = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "Tester"},
        {"role": "user", "content": q}
    ]
)
print(rsp.get("choices")[0]["message"]["content"])

