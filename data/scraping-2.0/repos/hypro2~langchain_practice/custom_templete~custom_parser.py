import os

from util import config_util

config = config_util.ConfigClsf().get_config()
openai_api_key = os.getenv('OPENAI_API_KEY', config['OPENAI']['API'])

from langchain.schema import BaseOutputParser


class CustomOutputParser(BaseOutputParser):

    def parse(self, text: str):
        """LLM 결과 물을 파싱"""
        return text

    def get_format_instructions(self) -> str:
        """LLM에 input으로 사용될 수 있는 예시 작성."""
        return (
            "Your response should be "
            "eg:"
        )

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def _type(self) -> str:
        return ""


