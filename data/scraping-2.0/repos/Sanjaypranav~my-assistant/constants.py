"""
------------------------------------
Constant file for the project
------------------------------------
"""
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate


DEFAULT_PROMPT : str = json.loads(open('data.json').read())[0]["prompt"]
# print(DEFAULT_PROMPT)
NAME: str = json.loads(open('data.json').read())[0]["name"]

with open("openai_api_key.txt", "r") as f:
    api_key = f.read()

LLM = ChatOpenAI(
    model = "gpt-4",
    openai_api_key=api_key
)


PROMPT = PromptTemplate(input_variables=["emotion"], template=DEFAULT_PROMPT)