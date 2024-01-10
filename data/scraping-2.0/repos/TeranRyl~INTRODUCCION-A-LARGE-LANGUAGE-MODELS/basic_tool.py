from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.tools import tool
import os
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone

# Pinecone and OpenAI Embedding initialization
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
embeddings = OpenAIEmbeddings()


@tool("SayHello", return_direct=True)
def say_hello(name: str) -> str:
    """ Answer when someone says hello"""
    return f"Hello {name}! My name is Sainapsis"


@tool("Search", return_direct=True)
def search(query: str, pinecone_index: str) -> list:
    """Consult the documents and generate a response"""
    # if you already have an index, you can load it like this
    docsearch = Pinecone.from_existing_index(pinecone_index, embeddings)
    docs = docsearch.similarity_search(query)
    return docs


def main():
    llm = ChatOpenAI(temperature=0)
    tools = [
        say_hello,
        search
    ]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent = AgentType.OPENAI_FUNCTIONS,
        verbose = True
    )
    print(agent.run("Hello! My name is Juan"))

    # print(text.read())
    pinecone_index = "sainapsis"
    pinecone.create_index(pinecone_index, dimension=1536, metric="cosine")

    """ Start of teacher's code fragment"""
    text = open("economia.txt", "r")
    data = Pinecone.from_texts(texts=text.readlines(), embedding=embeddings, index_name=pinecone_index)
    text = open("ingenieria-civil.txt", "r")
    data = Pinecone.from_texts(texts=text.readlines(), embedding=embeddings, index_name=pinecone_index)
    text = open("ingenieria-sistemas.txt", "r")
    data = Pinecone.from_texts(texts=text.readlines(), embedding=embeddings, index_name=pinecone_index)
    text = open("ingenieria-electrica.txt", "r")
    data = Pinecone.from_texts(texts=text.readlines(), embedding=embeddings, index_name=pinecone_index)
    text = open("ingenieria-industrial.txt", "r")
    data = Pinecone.from_texts(texts=text.readlines(), embedding=embeddings, index_name=pinecone_index)
    """ End of teacher's code fragment"""

    query = input("Por favor digite su consulta (FAQs de ECI): ")
    print(agent.run(search(query, pinecone_index)))


if __name__ == "__main__":
    main()


