from langchain.chat_models import ChatOpenAI


def get_agent(temperature=0.7):
    agent = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
    return agent
