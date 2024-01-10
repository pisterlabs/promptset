import os
from dotenv import load_dotenv
from llms.openia import OpenAI
from llms.hugging_face import HuggingFace
from enums.llm_source import LlmSource

class LlmFactory:
    def create(self):
        load_dotenv()
        llm_source = os.getenv("LLM_SOURCE")
        if llm_source == LlmSource.EXTERNAL.value:
            return  OpenAI()

        return HuggingFace()
