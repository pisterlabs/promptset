from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  AIMessagePromptTemplate,
  HumanMessagePromptTemplate,
)
from langchain.schema import (AIMessage, HumanMessage, SystemMessage)


def run_chat():
  chat = ChatOpenAI(temperature=0)

  while (True):
    text = input("Enter a message to be Socrates-d: ")
    messages = [
      SystemMessage(
        content="You are Socrates. Please identify any gaps in my thinking."),
      HumanMessage(content=text)
    ]
    print(chat(messages))
