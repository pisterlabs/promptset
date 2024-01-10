import os
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.llms import AzureOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from utils.wikipedia_answer import Wiki_QA


ENV_FILE = find_dotenv()
load_dotenv(ENV_FILE) # find_dotenv is called automatically

completion_llm = AzureOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base=os.environ["OPENAI_API_BASE"],
    deployment_name=os.environ["COMPLETION_ENGINE"]
    ) # type: ignore

chat_llm = AzureChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base=os.environ["OPENAI_API_BASE"],
    deployment_name=os.environ["CHAT_ENGINE"]
    ) # type: ignore

chat_llm.temperature = 0

messages = [
    SystemMessage(
        content="You are a helpful assistant."
    ),
    HumanMessage(
        content="Hi Jarvis, how are you today?"
    ),
    AIMessage(content="My name is Jarvis, how may I assist you today?"),
]


def ai_response(message: str):
    messages.append(HumanMessage(content=message))
    return chat_llm(messages=messages).content

qa = Wiki_QA(llm=completion_llm)

def answer_question(question: str):
    return qa.answer_question(question=question)


if __name__ == "__main__":
    # print(llm("Tell me a joke"))
    ai_response("Who was Donald Trump")

"""
If necessary adjust the model parameters
llm.temperature = 0.8
llm.top_p = 1
"""