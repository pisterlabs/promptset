from langchain import embeddings
from client.chains.vectorstores import get_vectorstore_from_docs
from client.chains.embeddings import get_embeddings
from client.chains.schema import Document
from client.chains.tools import get_all_tools

def get_docs_from_tools(tools):
    """
    Generates a list of Document objects from the given tools.

    Args:
        tools (List[Tool]): A list of Tool objects.

    Returns:
        List[Document]: A list of Document objects generated from the tools.
    """
    return [
        Document(page_content=t.description, metadata={"index": i})
        for i, t in enumerate(tools)
    ]

def get_retriever(k=4, model_name="intfloat/e5-large-v2"):
    """
    Returns a retriever object initialized with the specified parameters.

    Parameters:
        k (int): The number of nearest neighbors to retrieve.
        model_name (str): The name of the model to use for embeddings.

    Returns:
        Retriever: A retriever object.
    """
    all_tools = get_all_tools()
    docs = get_docs_from_tools(all_tools)
    embeddings = get_embeddings(model_name=model_name)
    vectordb = get_vectorstore_from_docs(docs, embeddings, dirpath="tools_db")
    return vectordb.as_retriever(search_kwargs={"k": k})

def get_relevant_docs_from_retriever(retriever, context):
    """
    Retrieves relevant documents from a retriever based on a given context.

    Args:
        retriever: The retriever object used to retrieve documents.
        context: The context used to retrieve relevant documents.

    Returns:
        The relevant documents retrieved from the retriever.
    """
    return retriever.get_relevant_documents(context)
