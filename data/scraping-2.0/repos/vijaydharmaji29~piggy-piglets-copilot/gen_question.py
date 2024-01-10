from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

open_ai_key = "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"

def ask_gen_question(question):

    chat = ChatOpenAI(temperature=0, openai_api_key=open_ai_key)

    question = question

    messages = [
        SystemMessage(
            content="You are a helpful assistant"
        ),
        HumanMessage(
            content=question
        )
    ]
    return chat(messages).content



