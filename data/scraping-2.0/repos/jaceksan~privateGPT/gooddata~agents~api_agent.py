import os
from enum import Enum
from pathlib import Path

import openai
from dotenv import load_dotenv
from gooddata_sdk import GoodDataSdk
from openapi_parser import parse
from openapi_parser.enumeration import OperationMethod
import streamlit as st

load_dotenv()


class ApiAgent:
    class Model(Enum):
        GPT_4 = "gpt-4"
        GPT_4_FUNC = "gpt-4-0613"
        GPT_3 = "gpt-3.5-turbo"
        GPT_3_FUNC = "gpt-3.5-turbo-0613"

    class AIMethod(Enum):
        RAW = "raw"
        FUNC = "functional"
        LANGCHAIN = "langchain"

    def __init__(self, workspace_id: str, open_ai_model: Model = Model.GPT_3_FUNC, method: AIMethod = AIMethod.FUNC) -> None:
        self.openAIModel = open_ai_model
        self.method = method

        load_dotenv()
        assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY not found in environment variables"
        openai.api_key = os.environ["OPENAI_API_KEY"]

        self.path_to_profiles = Path("gooddata/agents/profiles.yaml")
        self.sdk = GoodDataSdk.create_from_profile(profile="default", profiles_path=self.path_to_profiles)
        self.workspace_id = workspace_id
        self.unique_prefix = f"GOODDATA_PHOENIX"
        self.spec = get_api_spec()

    def get_open_ai_sys_msg(self) -> str:
        sys_msg = f"""
------------------------
Context:
------------------------
There are {self.unique_prefix} APIs consisting of API paths with descriptions.
Example: 
API path: /api/v1/entities/workspaces
API description: List all workspaces
        

Here is the full list of APIs:
-----------------------------------
"""

        for api_path in self.spec.paths:
            if api_path.url.startswith("/api/v1/entities"):
                print(f"------ {api_path.url} -----------")
                for operation in api_path.operations:
                    if operation.method == OperationMethod.GET:
                        print(f"------ {api_path.url} -----------")
                        sys_msg += f"""
API path: {api_path.url}
API description: {operation.summary}
"""
        sys_msg += f"""
If you are asked for anything, what relates to the above APIs descriptions, return the corresponding API path.
Example Question: list workspaces
Example answer: /api/v1/entities/workspaces

If the API path contains placeholder {{workspaceId}}, replace it with {self.workspace_id}.
If the API path contains placeholder {{dataSourceId}}, replace it with "demo" string.
"""

        return sys_msg

    def get_open_ai_raw_prompt(self, question: str) -> str:
        return f"""
Answer this question: "{question}" 

Context: {self.unique_prefix}
"""

    def ask_open_ai_raw(self, prompt: str) -> str:
        print(
            f"""Asking OpenAI.
              model: {self.openAIModel.value}
              method: {self.method.value}"""
        )

        with open("tmp/xxxx.txt", "w") as fp:
            fp.write(self.get_open_ai_sys_msg())
            fp.write(self.get_open_ai_raw_prompt(prompt))

        completion = openai.ChatCompletion.create(
            model=self.openAIModel.value,
            messages=[
                {"role": "system", "content": self.get_open_ai_sys_msg()},
                {"role": "user", "content": self.get_open_ai_raw_prompt(prompt)},
            ],
        )
        print(f"Tokens: {completion.usage}")

        return completion.choices[0].message.content

    def ask(self, question: str) -> str:
        return self.ask_open_ai_raw(question)


@st.cache_data
def get_api_spec():
    return parse("gooddata/open-api-spec.json", strict_enum=False)
