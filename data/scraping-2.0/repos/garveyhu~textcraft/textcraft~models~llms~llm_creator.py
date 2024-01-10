from textcraft.core.config import default_model, keys_openai, model_temperature
from textcraft.models.llms.ernie import Ernie
from textcraft.models.llms.openai import OpenAI
from textcraft.models.llms.qwen import Qwen
from textcraft.models.llms.spark import Spark


class LLMCreator:
    llms = {
        "text-davinci-003": OpenAI,
        "gpt-3.5-turbo": OpenAI,
        "gpt-3.5-turbo-16k": OpenAI,
        "gpt-3.5-turbo-1106": OpenAI,
        "gpt-4": OpenAI,
        "gpt-4-32k": OpenAI,
        "gpt-3.5-turbo-1106": OpenAI,
        "ernie-bot-turbo": Ernie,
        "ernie-bot-4.0": Ernie,
        "spark-v2": Spark,
        "spark-v3": Spark,
        "qwen-turbo": Qwen,
        "qwen-plus": Qwen,
    }

    @classmethod
    def create_llm(cls, llm_type=None):
        if llm_type is None:
            llm_type = default_model()

        llm_class = cls.llms.get(llm_type.lower())
        if not llm_class:
            raise ValueError(f"No LLM class found for type {llm_type}")

        if llm_class == OpenAI:
            openai_key = keys_openai()
            temperature = model_temperature()
            return OpenAI(model=llm_type.lower(), temperature=temperature, api_key=openai_key)

        return llm_class()
