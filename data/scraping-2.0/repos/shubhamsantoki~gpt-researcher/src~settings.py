import os
import pinecone

from langchain.retrievers.web_research import WebResearchRetriever


def connect_to_vectorstore():
    """code for connecting to pinecone database"""

    pinecone.init(
        api_key=os.environ.get("PINECONE_API_KEY"),
        environment=os.environ.get("PINECONE_ENVIORNMENT"),
    )
    return pinecone.Index(os.environ.get("PINECONE_INDEX"))


def settings():

    # Vectorstore
    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.memory import ConversationSummaryBufferMemory
    embeddings_model = OpenAIEmbeddings()
    index = connect_to_vectorstore()
    vectorstore = Pinecone(index, embeddings_model.embed_query, "web_researcher")

    # LLM
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0, streaming=True)

    # Search
    from langchain.utilities import GoogleSearchAPIWrapper

    search = GoogleSearchAPIWrapper()

    # Initialize
    web_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore, llm=llm, search=search, num_search_results=1
    )

    #memory for retriever
    memory = ConversationSummaryBufferMemory(llm=llm, input_key='question', output_key='answer', return_messages=True)


    return web_retriever, llm, memory
