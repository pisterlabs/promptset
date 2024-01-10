import os
from enum import Enum
from time import time
from typing import Optional

import openai
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI

from gooddata.agents.sdk_wrapper import GoodDataSdkWrapper


class AIMethod(Enum):
    RAW = "raw"
    FUNC = "functional"
    LANGCHAIN = "langchain"


# TODO - limit the list of the supported models?
class AIModel(Enum):
    GPT_35 = "gpt-3.5-turbo-0613"
    GPT_4 = "gpt-4-turbo-0613"


class GoodDataOpenAICommon:
    def __init__(
        self,
        openai_model: str = "gpt-3.5-turbo-0613",
        openai_api_key: Optional[str] = None,
        openai_organization: Optional[str] = None,
        temperature: int = 0,
        workspace_id: str = None,
    ) -> None:
        load_dotenv()
        self.openai_model = openai_model
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_organization = openai_organization or os.getenv("OPENAI_ORGANIZATION")
        self.temperature = temperature
        self.workspace_id = workspace_id

        self.unique_prefix = f"GOODDATA_PHOENIX::{self.workspace_id}"
        self.gd_sdk = GoodDataSdkWrapper()

        self.chain = self.get_conversation_chain()

    def log_openai_start(self, method: AIMethod = AIMethod.RAW):
        print(
            f"""Asking OpenAI.
              model: {self.openai_model}
              method: {method.value}"""
        )

    @property
    def openai_kwargs(self) -> dict:
        kwargs = {
            "temperature": self.temperature,
            "model_name": self.openai_model,
        }
        if self.openai_api_key:
            kwargs["openai_api_key"] = self.openai_api_key
            kwargs["openai_organization"] = self.openai_organization
        return kwargs

    @property
    def openai_chat_credential_kwargs(self) -> dict:
        return {
            "api_key": self.openai_api_key,
            "organization": self.openai_organization,
        }

    def get_chat_llm_model(self):
        return ChatOpenAI(**self.openai_kwargs)

    def get_conversation_chain(self) -> ConversationChain:
        llm = ChatOpenAI(**self.openai_kwargs)

        chain = ConversationChain(llm=llm)
        return chain

    def ask_question(self, request: str) -> str:
        start = time()
        self.log_openai_start()
        chain = self.get_conversation_chain()
        result = chain.run(input=request)
        duration = int((time() - start) * 1000)
        print(f"OpenAI query finished duration={duration}")
        return result

    def ask_chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        functions: Optional[list[dict]] = None,
        function_name: Optional[str] = None,
    ):
        start = time()
        method = AIMethod.FUNC if functions else AIMethod.FUNC
        self.log_openai_start(method)
        kwargs = {
            **self.openai_chat_credential_kwargs,
            "model": self.openai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if functions:
            kwargs["functions"] = functions
        if function_name:
            kwargs["function_call"] = {"name": function_name}

        completion = openai.ChatCompletion.create(**kwargs)
        print(f"Tokens: {completion.usage}")
        duration = int((time() - start) * 1000)
        print(f"OpenAI query finished duration={duration}")
        return completion
