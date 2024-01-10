import textwrap
import typing

import openai

from .__log import log
from .__type import GenerateProvider

openai.api_base = 'https://api.chatanywhere.com.cn/v1'
openai.api_key = 'sk-0PfcSdT723UR44igwVxvEWvLoZJgi0FJyZWy0WCCATp5ka2a'


class ChatGPT3Provider(GenerateProvider):
    def __init__(self):
        self.prompt = textwrap.dedent('''
            You are a Python programmer, and I will give you some Python function signatures. You need to return the specific implementation of the function for me, and your return results should not contain any explanatory text.
        ''')[1:]
        log.info('prompt: %s', self.prompt)

    def generate(self, func: typing.Callable, define: str) -> str:
        log.info('define: %s', define)
        result = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=[
            {'role': 'system', 'content': self.prompt},
            {'role': 'user', 'content': define}
        ])['choices'][0]['message']['content']
        log.info('result: %s', result)
        return result[10:-3]
