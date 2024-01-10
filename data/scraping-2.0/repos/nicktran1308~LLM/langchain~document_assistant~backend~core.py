# ------------------------------- Import Libraries -------------------------------
import os
from typing import Any, Dict, List
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
import pinecone

# ------------------------------- Pinecone Initialization -------------------------------
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)
INDEX_NAME = "langchain-doc-index"

# ------------------------------- Conversational Retrieval with LLM -------------------------------
def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    """
    Conduct a conversational retrieval using an LLM and the Pinecone vector store.
    """
    # Initialize embeddings and document search 
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    docsearch = Pinecone.from_existing_index(
        embedding=embeddings,
        index_name=INDEX_NAME,
    )

    # Set up chat model
    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
    )

    # Initialize and execute the conversational retrieval chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )
    return qa({"question": query, "chat_history": chat_history})

