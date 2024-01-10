# ensure that the OPENAI_API_KEY is set in the environment
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# create a chat model
# can be editied to gpt4
chat = ChatOpenAI(model_name='gpt-3.5-turbo')

print(chat)

# from langchain.vectorstores import FAISS
# from langchain.embeddings.openai import OpenAIEmbeddings

# # load the vector store database
# embeddings = OpenAIEmbeddings()
# db = FAISS.load_local("faiss_index", embeddings)

llm = OpenAI()

# define a chat request function so that it can be exported to flask app


def chatRequest(query: str):
    response = chat([HumanMessage(content=query)])
    return response.content


# def chatRequest():

# show available engines
