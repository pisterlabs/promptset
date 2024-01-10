import json
import re
from pathlib import Path

import pandas as pd
from gooddata_sdk import Attribute, ObjId, SimpleMetric
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

from gooddata.agents.common import AIMethod, GoodDataOpenAICommon
from gooddata.tools import TMP_DIR, create_dir


class ReportAgent(GoodDataOpenAICommon):
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
        exdef = json.loads(result)
        return exdef

    def get_langchain_query(self, question: str) -> str:
        return f"""Create "{question}" from metrics and attributes in {self.unique_prefix} as Execution Definition.
            Write only the .json without any explanation.
            This means, that you always start with '{' and end with '}'."""

    @staticmethod
    def get_open_ai_sys_msg() -> str:
        return """
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
        metrics:{self.gd_sdk.metrics_string(self.workspace_id)}
        attributes:{self.gd_sdk.attributes_string(self.workspace_id)}
        \"\"\"
        """

    def get_open_ai_raw_prompt(self, question: str) -> str:
        return f"""
        Create "{question}" in {self.unique_prefix} as ExecutionDefinition json.

        context:\"\"\"
        This is {self.unique_prefix}:
        metrics:{self.gd_sdk.metrics_string(self.workspace_id)}
        attributes:{self.gd_sdk.attributes_string(self.workspace_id)}
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

    def ask_open_ai_raw(self, question: str) -> str:
        completion = self.ask_chat_completion(
            system_prompt=self.get_open_ai_sys_msg(),
            user_prompt=self.get_open_ai_raw_prompt(question),
        )
        return completion.choices[0].message.content

    def ask_func_open_ai(self, question: str) -> str:
        completion = self.ask_chat_completion(
            system_prompt=self.get_open_ai_fnc_info(),
            user_prompt=self.get_functions_prompt(question),
            functions=[self.get_execdef_fnc()],
            function_name="ExecutionDefinition",
        )
        return completion.choices[0].message.function_call.arguments

    def get_workspace_loader(self, file_dir: Path, file_name: str):
        create_dir(file_dir)
        file_path = file_dir / file_name
        with open(file_path, "w") as fp:
            fp.write(f"{self.get_open_ai_sys_msg()}\n")
            ws_content = self.gd_sdk.sdk.catalog_workspace_content.get_full_catalog(self.workspace_id)
            metrics_list = [[m.id, m.title] for m in ws_content.metrics]
            attribute_list = [[a.id, a.title] for a in ws_content.attributes]
            fp.write(f"Metrics: {str(metrics_list)}\n")
            fp.write(f"Attributes: {str(attribute_list)}\n")
        return TextLoader(str(file_path))

    def ask_langchain_open_ai(self, question: str) -> str:
        loader = self.get_workspace_loader(TMP_DIR, f"report_agent_loader_{self.workspace_id}.txt")
        index = VectorstoreIndexCreator().from_loaders([loader])

        chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model=self.openai_model),
            retriever=index.vectorstore.as_retriever(),
        )

        return chain.run(self.get_langchain_query(question))

    def ask(self, method: AIMethod, question: str) -> str:
        match method:
            case AIMethod.FUNC:
                return self.ask_func_open_ai(question)
            case AIMethod.RAW:
                return self.ask_open_ai_raw(question)
            case AIMethod.LANGCHAIN:
                return self.ask_langchain_open_ai(question)

            case _:
                print("No method found, defaulting to RAW")
                return self.ask_open_ai_raw(question)

    def execute_report(self, answer: str) -> tuple[pd.DataFrame, list, list]:
        """Get Pandas data frame the generated ExecutionDefinition

        Args:
            answer (str):
                Answer from the OpenAI agent, can be either a valid .json or
                a written text containing the valid .json
        """

        exdef = self.answer_to_json(answer)

        frames = self.gd_sdk.pandas.data_frames(self.workspace_id)

        attributes = {attr: Attribute(local_id=attr, label=attr) for attr in exdef["attributes"]}
        metrics = {metr: SimpleMetric(local_id=metr, item=ObjId(metr, type="metric")) for metr in exdef["metrics"]}
        df = frames.for_items(items={**attributes, **metrics}, auto_index=False)
        return df, exdef["attributes"], exdef["metrics"]

    def process(self, method: AIMethod, question: str) -> tuple[pd.DataFrame, list, list]:
        """
        Method orchestrating the whole process
        :return:
        """
        answer = self.ask(method, question)
        return self.execute_report(answer)
