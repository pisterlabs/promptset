from langchain.memory import ConversationSummaryBufferMemory
from langchain.llms import OpenAI


def CSBM():
    return ConversationSummaryBufferMemory(llm=OpenAI(temperature=0),
                                           max_token_limit=2000,
                                           memory_key="chat_history",
                                           ai_prefix="Chatbot PDF",
                                           human_prefix="Usuario",
                                           return_messages=True)
