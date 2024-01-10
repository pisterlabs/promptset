import openai
import os
import ssl

import warnings
warnings.filterwarnings("ignore")
# 只需要在python里设置代理即可
#os.environ['HTTP_PROXY'] = 'http://202.79.168.14:6666'
#os.environ['HTTPS_PROXY'] = 'http://202.79.168.14:6666'
#openai没办法使用全局代理，在lib\site-packages\openai\api_requestor.py文件里添加代理即可

# ssl._create_default_https_context = ssl._create_unverified_context
# openai.api_key = '*********'
# # openai.proxy = 'http://202.79.168.46:6666'
# # openai.verify_ssl_certs = False
# openai.api_base='https://202.79.168.46/v1'
# def test_openai(string):
#     completion = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[{"role": "user", "content": string}]
#     )
#     return completion['choices'][0]['message']['content'].strip()
#
# res=test_openai("巩义在哪里？")
#
# print(res)

class GptApi:
    def __init__(self):
        self.api_key='***********'
        openai.api_key = self.api_key
        openai.api_base='https://202.79.168.46/v1'

    def test_openai(self,string):
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": string}]
        )
        return completion['choices'][0]['message']['content'].strip()



if __name__ == '__main__':
    test=GptApi()
    res=test.test_openai('你是谁？')
    print(res)
