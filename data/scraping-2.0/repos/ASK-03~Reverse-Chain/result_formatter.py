import os
from configparser import ConfigParser

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

from modules import ReverseChainBaseClass

import warnings

warnings.filterwarnings("ignore")

config = ConfigParser()
config.read("config.ini")

OPENAI_SECRET_KEY = config["openai"]["secret_key"]
MODEL = config["openai"]["model"]
TEMPERATURE = float(config["openai"]["temperature"])

os.environ["OPENAI_API_KEY"] = OPENAI_SECRET_KEY


class ResultFormatterBaseClass(ReverseChainBaseClass):
    def __init__(self, model, temperature) -> None:
        super(ResultFormatterBaseClass, self).__init__(model, temperature)

    def _format(self) -> None:
        raise NotImplementedError


class ResultFormatter(ResultFormatterBaseClass):
    def __init__(self, model, temperature):
        super(ResultFormatter, self).__init__(model, temperature)
        self.template = """
        on being given the context, use the given context of the tools and arguments and output the response.
        The output should be a list of JSONs conforming following jsonschema:

        Unset
        {{
            "type": "array",
            "items": {{
                "type": "object",
                "properties": {{
                    "tool_name": {{ "type": "string" }},
                    "arguments": {{
                        "type": "array",
                        "items": {{
                            "type": "object",
                            "properties": {{
                                "argument_name": {{ "type": "string" }},
                                "argument_value": {{ "type": "string" }}
                            }},
                            "required": ["argument_name", "argument_value"]
                        }}   
                    }}
                }},
                "required": ["tool_name", "arguments"]
            }}
        }}

        In the above json schema, replace these values accordingly.
        Donot use those arguments whose value is RequiredFalse
        Context:
        {context}

        Result: 
        """

    def get_prompt(self, context: str) -> str:
        prompt = PromptTemplate(input_variables=["context"], template=self.template)
        return prompt.format(context=context)

    def _format(self, context):
        prompt = self.get_prompt(context=context)
        response = self.llm(prompt)
        return response

    def run(self, context):
        formatted_result = self._format(context)
        return formatted_result


# if __name__ == "__main__":
# context = """
# {{
#     {{
#         'sequence_no': 0,
#         'api_name': 'create_actionable_tasks_from_text',
#         'input_arguments': {{
#             'text': '<Transcript>'
#         }}
#         'output': {{
#             "tasks": ['A123', 'B123']
#         }}
#     }},
#     {{
#         'sequence_no': 1,
#         'api_name': 'get_sprint_id',
#         'input_arguments': {{}},
#         'output': {{
#             "sprint_id": 'SPRINT123'
#         }}

#     }},
#     {{
#         'sequence_no': 2
#         'api_name': 'add_work_items_to_sprint'
#         'input_arguments': {{
#             'work_ids': ['A123', 'A234'],
#             'sprint_id': "SPRINT123"
#         }},
#         'output': {}
#     }}
# }}
# """

# # result_formatter = ResultFormatter(MODEL, TEMPERATURE)
# # formatted_result = result_formatter._format(context)
# # print(formatted_result)
