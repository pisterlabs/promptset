from langchain.chat_models import ChatOpenAI


def get_llm(apikey):
    return ChatOpenAI(verbose=True, model="gpt-4", api_key=apikey, streaming=True)


print(get_llm(1234))
