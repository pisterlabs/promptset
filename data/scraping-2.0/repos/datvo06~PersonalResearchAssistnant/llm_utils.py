from langchain.chat_models import ChatOpenAI


def get_gpt4_llm():
    return ChatOpenAI(model_name = "gpt-4")


def get_gpt35_turbo_llm():
    return ChatOpenAI(model_name = "gpt-3.5-turbo")
