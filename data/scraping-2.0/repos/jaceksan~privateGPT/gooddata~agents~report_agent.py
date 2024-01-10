import os
from enum import Enum
from pathlib import Path

import openai
import pandas as pd
from dotenv import load_dotenv
from gooddata_sdk import Attribute, ExecutionDefinition, GoodDataSdk, ObjId, SimpleMetric
from gooddata_pandas import GoodPandas
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

import json
import re

load_dotenv()


class ReportAgent:
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
        self.metrics_string = str(
            [
                [metric.id, metric.title]
                for metric in self.sdk.catalog_workspace_content.get_metrics_catalog(workspace_id=self.workspace_id)
            ]
        )
        self.attribute_string = str(
            [
                [attr.id, attr.title]
                for attr in self.sdk.catalog_workspace_content.get_attributes_catalog(workspace_id=self.workspace_id)
            ]
        )
        self.unique_prefix = f"GOODDATA_PHOENIX::{self.workspace_id}"

    @staticmethod
    def answer_to_json(answer: str) -> dict:
        """Transform answer to dict, no matter the format.

        This is a bulldozer of a function.

        Args:
            answer (str): Answer from the OpenAI agent.

        Returns:
            dict: Parsed json response
        """
        regex = r"\{(?:[^{}]|)*\}"
        matches = re.search(regex, answer)
        result = matches.group()
        if result == "":
            raise Exception("Did not contain .json!")
        print(result)
        exdef = json.loads(result)
        return exdef

    def execute_report(self, workspace_id: str, answer: str) -> tuple[pd.DataFrame, list, list]:
        """Get Pandas data frame the generated ExecutionDefinition

        Args:
            workspace_id (str):
                GoodData workspace ID, against which we want to execute a report
            answer (str):
                Answer from the OpenAI agent, can be either a valid .json or
                a written text containing the valid .json
        """

        exdef = self.answer_to_json(answer)

        gp = GoodPandas.create_from_profile(profile="default", profiles_path=self.path_to_profiles)
        frames = gp.data_frames(workspace_id)

        # attributes = [Attribute(local_id=attr, label=attr) for attr in exdef["attributes"]]
        # metrics = [SimpleMetric(local_id=metr, item=ObjId(metr, type="metric")) for metr in exdef["metrics"]]
        # exec_def = ExecutionDefinition(
        #     attributes=attributes,
        #     metrics=metrics,
        #     dimensions=[[attr for attr in exdef["attributes"]], ["measureGroup"]],
        #     filters=[],
        # )
        # df, df_metadata = frames.for_exec_def(exec_def=exec_def)
        attributes = {attr: Attribute(local_id=attr, label=attr) for attr in exdef["attributes"]}
        metrics = {metr: SimpleMetric(local_id=metr, item=ObjId(metr, type="metric")) for metr in exdef["metrics"]}
        df = frames.for_items(items={**attributes, **metrics}, auto_index=False)
        return df, exdef["attributes"], exdef["metrics"]

    def get_langchain_query(self, question: str) -> str:
        return f"""Create "{question}" from metrics and attributes in {self.unique_prefix} as Execution Definition.
            Write only the .json without any explanation.
            This means, that you always start with '{' and end with '}'."""

    @staticmethod
    def get_open_ai_sys_msg() -> str:
        return f"""
            You create ExecutionDefinition and always write only the .json without any explanation.
            This means, that you always start with "{{" and end with "}}".

            To create an ExecutionDefinition, you need to provide .json in such structure:
            {{
                "attributes": ["string"],
                "metrics": ["string"],
            }}

            Where:
            Attributes are a list containing strings, representing the identifier of the attribute from the workspace.
            Metrics are a list containing strings, representing the identifier of the metrics from the workspace.
            """

    def get_open_ai_fnc_info(self) -> str:
        return f"""
        Whenever you create a new ExecutionDefinition, you always work upon {self.unique_prefix}, defined as:
        metrics:{self.metrics_string}
        attributes:{self.attribute_string}
        \"\"\"
        """

    def get_open_ai_raw_prompt(self, question: str) -> str:
        return f"""
        Create "{question}" in {self.unique_prefix} as ExecutionDefinition json.

        context:\"\"\"
        This is {self.unique_prefix}:
        metrics:{self.metrics_string}
        attributes:{self.attribute_string}
        \"\"\"
        """

    def get_functions_prompt(self, question: str) -> str:
        """Prompt for Function Calls from OpenAI.

        Returns:
            str: OpenAI function call prompt
        """
        return f"""
        Create "{question}" as ExecutionDefinition, strictly from the metrics and attributes in {self.unique_prefix}.
        """

    @staticmethod
    def get_execdef_fnc() -> dict:
        """ExecutionDefinition function definition for OpenAI function calls

        Returns:
            str: Definition of the ExecDef
        """
        return {
            "name": "ExecutionDefinition",
            "description": "Create ExecutionDefinition for data visualization",
            "parameters": {
                "type": "object",
                "properties": {
                    "attributes": {
                        "type": "array",
                        "items": {"type": "string", "description": "local_id of an attribute"},
                        "description": "List of local_id of attributes to be used in the visualization",
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string", "description": "local_id of a metric"},
                        "description": "List of local_id of metrics to be used in the visualization",
                    },
                },
                "required": ["attributes", "metrics"],
            },
        }

    def ask_open_ai_raw(self, prompt: str) -> str:
        print(
            f"""Asking OpenAI.
              model: {self.openAIModel.value}
              method: {self.method.value}"""
        )

        completion = openai.ChatCompletion.create(
            model=self.openAIModel.value,
            messages=[
                {"role": "system", "content": self.get_open_ai_sys_msg()},
                {"role": "user", "content": self.get_open_ai_raw_prompt(prompt)},
            ],
        )
        print(f"Tokens: {completion.usage}")

        return completion.choices[0].message.content

    def ask_func_open_ai(self, question: str) -> str:
        print(
            f"""Asking OpenAI.
              model: {self.openAIModel.value}
              method: {self.method.value}"""
        )

        completion = openai.ChatCompletion.create(
            model=self.openAIModel.value,
            messages=[
                {"role": "system", "content": self.get_open_ai_fnc_info()},
                {"role": "user", "content": self.get_functions_prompt(question)},
            ],
            functions=[self.get_execdef_fnc()],
            function_call={"name": "ExecutionDefinition"},
        )

        print(f"Tokens: {completion.usage}")
        return completion.choices[0].message.function_call.arguments

    def get_workspace_loader(self, workspace_id: str, file_dir: str, file_name: str):
        file_path = Path(file_dir) / file_name
        if not os.path.isdir(file_dir):
            os.mkdir(file_dir)
        with open(file_path, "w") as fp:
            fp.write(f"{self.get_open_ai_sys_msg()}\n")
            ws_content = self.sdk.catalog_workspace_content.get_full_catalog(workspace_id)
            metrics_list = [[m.id, m.title] for m in ws_content.metrics]
            attribute_list = [[a.id, a.title] for a in ws_content.attributes]
            fp.write(f"Metrics: {str(metrics_list)}\n")
            fp.write(f"Attributes: {str(attribute_list)}\n")
        return TextLoader(str(file_path))

    def ask_langchain_open_ai(self, question: str) -> str:
        print(
            f"""Asking OpenAI.
              model: {self.openAIModel.value}
              method: {self.method.value}"""
        )

        loader = self.get_workspace_loader(self.workspace_id, "tmp", f"report_agent_{self.workspace_id}.txt")
        index = VectorstoreIndexCreator().from_loaders([loader])

        chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model=self.openAIModel.value),
            retriever=index.vectorstore.as_retriever(),
        )

        return chain.run(self.get_langchain_query(question))

    def ask(self, question: str) -> str:
        match self.method:
            case self.AIMethod.FUNC:
                return self.ask_func_open_ai(question)
            case self.AIMethod.RAW:
                return self.ask_open_ai_raw(question)
            case self.AIMethod.LANGCHAIN:
                return self.ask_langchain_open_ai(question)

            case _:
                print("No method found, defaulting to RAW")
                return self.ask_open_ai_raw(question)
