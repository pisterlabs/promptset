# flake8: noqa
from langchain.prompts.prompt import PromptTemplate

_DEFAULT_TEMPLATE = """Given an input question, first select the correct API endpoint to run, then write python code that will execute the API endpoint according to the input.

You have access to this API:
{api_info}

for the API use the enviroment variables URL and BEARER_TOKEN

Use the following format:

Question: "Question here"
pythonCode: "Valid python code to run"
End.

You have access to the following resources:

{resources}

Question: {input}
pythonCode:"""

PROMPT = PromptTemplate(
    input_variables=["input", "api_info", "resources"], template=_DEFAULT_TEMPLATE
)
