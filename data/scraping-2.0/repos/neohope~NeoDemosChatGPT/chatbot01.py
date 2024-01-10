#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai
import yaml


'''
模拟聊天客服，同样的问题，回答多次，会得到不同答案
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


def get_response(prompt, temperature = 1.0):
    completions = openai.Completion.create (
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=temperature,
    )
    message = completions.choices[0].text
    return message


if __name__ == '__main__':
    get_api_key()
    prompt = '请你用朋友的语气回复给到客户，并称他为“亲”，他的订单已经发货在路上了，预计在3天之内会送达，订单号2021AEDG，我们很抱歉因为天气的原因物流时间比原来长，感谢他选购我们的商品。'
    print(get_response(prompt))
    print(get_response(prompt))
    print(get_response(prompt))

'''
亲,你的订单2021AEDG，我们已经发货，我们估计这份订单会在3天之内送达，但是由于天气的原因，物流时间比事先预期的可能会更长，我们深感抱歉。谢谢你的选购！
亲，您的订单（2021AEDG）已经发货，预计在三天内会到达，因天气原因导致物流时间可能比平时长，还请您耐心等待。非常感谢你选购我们的商品。如果你有其他问题欢迎随时联系我们，感谢支持！
亲，您好！感谢您对我们产品的信任，您的订单2021AEDG已发货，延误是因为天气原因，估计会在3天内送达，如果有什么疑问可以随时联系我们。再次感谢您的支持！
'''