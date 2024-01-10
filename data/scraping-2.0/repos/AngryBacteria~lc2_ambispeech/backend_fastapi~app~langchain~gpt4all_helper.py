from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain.llms.gpt4all import GPT4All
from langchain_core.language_models import BaseLanguageModel

from app.data.data import GenericLangChainModel
from app.utils.logging_util import logger


class GPT4AllHelper(GenericLangChainModel):
    _instance = None
    llm = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(GPT4AllHelper, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        load_dotenv()
        model_path = os.getenv("GPT4ALL_MODEL_PATH")
        if os.getenv("GPT4ALL_MODEL_PATH") is None:
            logger.warning(
                f"Model path not found: {model_path}. The GPT4All LLM will not be functional until a valid path is "
                f"provided in .env File."
            )
        else:
            self.llm = GPT4All(model=os.getenv("GPT4ALL_MODEL_PATH"))
            logger.info("Created GPT4AllHelper")

        self._initialized = True

    def get_llm(self):
        if self.llm is None:
            raise RuntimeError(
                "GPT4All LLM is not properly initialized. Please check the model path."
            )
        return self.llm
