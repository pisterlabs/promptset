from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools import Tool
from langchain.chat_models import ChatOpenAI
import pinecone
from langchain.chains import ConversationalRetrievalChain

def search_pinecone_codebase(question):
    """
    Search in the Pinecone codebase for answers to your questions.

    Parameters:
        question (str): The question to ask.

    Returns:
        str: The answer to your question.    
    """
    
    # Initialize Pinecone
    pinecone.init(
        api_key="797c1452-d615-40b6-bc63-56bba3fca7db",
        environment="asia-southeast1-gcp-free"
    )

    # Define the index
    index_name = "codebase-test"
    index = pinecone.Index(index_name=index_name)

    # Define the retriever
    retriever = Pinecone(
        index,
        OpenAIEmbeddings().embed_query,
        "text"
    ).as_retriever()

    # Define the chain
    chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo"), retriever=retriever)

    return chain({"question": question, "chat_history": []})["answer"]

def get_pinecone_codebase_tool():
    """
    Create a custom tool based on the search_pinecone_codebase function.
    
    Returns:
        Tool: The custom tool.
    """
    search_codebase_tool = Tool.from_function(
        func=search_pinecone_codebase,
        name="Search Codebase",
        description="Search the codebase for answers to your questions."
    )

    return search_codebase_tool
