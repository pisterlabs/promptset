import json
import os

import openai
import requests

from config import open_api_key, open_api_organization_id

# export https_proxy=http://127.0.0.1:33210 http_proxy=http://127.0.0.1:33210 all_proxy=socks5://127.0.0.1:33211
# 这里写代理的ip及端口
# proxy = '127.0.0.1:33210'
# proxies = {
#     'http': 'http://' + proxy,
#     'https'
#     : 'https://' + proxy
# }

header = {
    "Content-Type": "application/json",
    "Authorization": "Bearer {}".format(open_api_key)
}

openai.api_key = open_api_key
openai.organization = open_api_organization_id


def chat_gpt_reply(message):
    """
    调用chatgpt接口回复
    :param message:
    :return:
    """
    data = {
        "model": "gpt-3.5-turbo",  # text-davinci-003	 gpt-3.5-turbo
        "messages": [{"role": "user", "content": message}]
    }
    res = requests.post('https://api.openai.com/v1/chat/completions', headers=header, data=json.dumps(data))
    if res.status_code == 200:
        messages = json.loads(res.content).get('choices')
        reply_meg = []
        for me in messages:
            reply_meg.append(me.get('message').get('content'))
        return reply_meg[0].strip() or None


def create_image(key_word):
    """
    使用chatgpt创建图片
    """
    response = openai.Image.create(
        prompt=key_word,
        n=1,
        size="1024x1024"
    )
    image_url = response['data'][0]['url']
    return image_url


def list_models():
    """
    列出所有gpt 模型
    :return:
    """
    print(openai.Model.list())
    return openai.Model.list()


if __name__ == "__main__":
    print(chat_gpt_reply("2008年奥运会举办地点"))
    create_image("重庆山城")
    list_models()
