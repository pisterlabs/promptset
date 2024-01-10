from langchain.chat_models import ChatOpenAI


def chat_openai():
    return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")