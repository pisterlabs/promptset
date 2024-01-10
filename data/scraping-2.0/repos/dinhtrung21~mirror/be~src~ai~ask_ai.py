import os

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# check necessary OS envs
assert os.getenv("OPENAI_API_TOKEN"), "env 'OPENAI_API_TOKEN' must be specified"

chat = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_TOKEN"), model="gpt-4")


def get_answer_chatGPT(msg: str) -> str:
    messages = [HumanMessage(content=msg)]
    out = chat(messages).content

    return out


def get_answer_t5(msg: str) -> str:
    # NOTE: Will implement this later
    pass
