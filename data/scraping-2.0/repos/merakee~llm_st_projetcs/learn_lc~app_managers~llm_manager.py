# pyhton
import os
import re
from enum import Enum
from dotenv import load_dotenv

# local
from app_managers.auth_manager import APIType
from app_managers.auth_manager import APIKey

# langchain
# LLM
from langchain.llms import OpenAI
# from langchain.llms import ChatOpenAI
# from langchain.llms.fake import FakeListLLM
from langchain.llms import HuggingFaceHub

# implementation


class ChatLLMManager:
    def __init__(self, llm):
        self.llm = llm

    @staticmethod
    def get_openai_chatllm(api_key, temperature=0.7, model_name="gpt-3.5-turbo"):
        # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        # model_name = "gpt-3.5-turbo"
        llm = ChatOpenAI(model_name=model_name, temperature=temperature,
                         openai_api_key=api_key.api_token)
        return llm


class LLMManager:
    def __init__(self, llm):
        self.llm = llm

    @staticmethod
    def get_openai_llm(api_key, temperature=0.7, model_name="gpt-3.5-turbo-instruct"):
        # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        # model_name = "gpt-3.5-turbo"
        llm = OpenAI(model_name=model_name, temperature=temperature,
                     openai_api_key=api_key.api_token)
        return llm

    def get_hf_llm(api_key, model_name="google/flan-t5-xxl", temperature=0.7, max_length=64):
        llm = HuggingFaceHub(
            repo_id=model_name, huggingfacehub_api_token=api_key.api_token, model_kwargs={"temperature": temperature, "max_length": max_length})
        return llm

    def get_llm(api_key, temperature=0.7, model_name=None):
        if api_key.api_type == APIType.OpenAI:
            if model_name:
                return LLMManager.get_openai_llm(api_key=api_key, temperature=temperature, model_name=model_name)
            else:
                return LLMManager.get_openai_llm(api_key=api_key, temperature=temperature)
        elif api_key.api_type == APIType.HuggingFace:
            if model_name:
                return LLMManager.get_hf_llm(api_key=api_key, temperature=temperature, model_name=model_name)
            else:
                return LLMManager.get_hf_llm(api_key=api_key, temperature=temperature)
        else:
            return None


class OpenAIModelInfo:
    @staticmethod
    def get_openai_llm_models_info():
        # https://platform.openai.com/docs/models/overview
        # https://openai.com/pricing#language-models
        openai_info = """
        gpt-4: More capable than any GPT-3.5 model, able to do more complex tasks, and optimized for chat. Will be updated with our latest model iteration 2 weeks after it is released.:	8,192 tokens:	Up to Sep 2021\n
        gpt-4-0613:	Snapshot of gpt-4 from June 13th 2023 with function calling data. Unlike gpt-4, this model will not receive updates, and will be deprecated 3 months after a new version is released.:	8,192 tokens:Up to Sep 2021\n
        gpt-4-32k:	Same capabilities as the standard gpt-4 mode but with 4x the context length. Will be updated with our latest model iteration.:	32,768 tokens:	Up to Sep 2021\n
        gpt-4-32k-0613:	Snapshot of gpt-4-32 from June 13th 2023. Unlike gpt-4-32k, this model will not receive updates, and will be deprecated 3 months after a new version is released.	:32,768 tokens	:Up to Sep 2021\n
        gpt-3.5-turbo:	Most capable GPT-3.5 model and optimized for chat at 1/10th the cost of text-davinci-003. Will be updated with our latest model iteration 2 weeks after it is released.:	4,097 tokens:	Up to Sep 2021\n
        gpt-3.5-turbo-16k:	Same capabilities as the standard gpt-3.5-turbo model but with 4 times the context.:	16,385 tokens	:Up to Sep 2021\n
        gpt-3.5-turbo-instruct:	Similar capabilities as text-davinci-003 but compatible with legacy Completions endpoint and not Chat Completions.	:4,097 tokens:	Up to Sep 2021\n
        gpt-3.5-turbo-0613:	Snapshot of gpt-3.5-turbo from June 13th 2023 with function calling data. Unlike gpt-3.5-turbo, this model will not receive updates, and will be deprecated 3 months after a new version is released.:	4,097 tokens:	Up to Sep 2021\n
        gpt-3.5-turbo-16k-0613:	Snapshot of gpt-3.5-turbo-16k from June 13th 2023. Unlike gpt-3.5-turbo-16k, this model will not receive updates, and will be deprecated 3 months after a new version is released.:	16,385 tokens	:Up to Sep 2021\n
        """
        return openai_info

    def get_openai_llm_models():
        info = OpenAIModelInfo.get_openai_llm_models_info()

        models = []
        for linet in info.split("\n"):
            if linet:
                model = linet.split(":")[0].strip()
                if model:
                    models.append(model)
        return models


class HuggingFaceModelInfo:
    @staticmethod
    def get_openai_llm_models_info():
        # https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
        # https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads
        pass
