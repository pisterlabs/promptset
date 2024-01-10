import random

from infrastructure.openai_client import OpenAIClient
from settings import settings


class AITextGenerator(object):
    def __init__(self):
        self.openai_client = OpenAIClient()

    def generate(self, trend_topic: str) -> str:
        # 文章を生成するための書き出しをランダムに取り出す
        prompt_template = random.choice(settings.openai_prompt_format_list)
        prompt = prompt_template.format(trend_topic)
        # 文章を生成する
        created_text = self.openai_client.create(prompt)
        return f'{prompt}{created_text}'
