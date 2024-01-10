import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

def init_pinecone():
    """
    Initialize Pinecone service.

    This function initializes the Pinecone service using the API key and environment variable.
    """
    pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENVIRONMENT_REGION"])
    
def ingest_docs() -> None:
    """
    Process and send Langchain documents to Pinecone.

    This function performs a series of operations to add Langchain documents to the Pinecone vector store.
    Documents are loaded using ReadTheDocsLoader, then split into texts, and finally sent to Pinecone with updated
    source URLs.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    init_pinecone()

    # Load documents
    loader = ReadTheDocsLoader(path="langchain-docs/langchain-docs/api.python.langchain.com/en/latest", encoding="utf-8")
    raw_docs = loader.load()
    print("Loaded {} documents".format(len(raw_docs)))

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", "\r\n", " ", ""])
    documents = text_splitter.split_documents(documents=raw_docs)
    print("Split {} documents".format(len(documents)))

    # Update document sources
    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print("Going to write {} documents".format(len(documents)))

    # Create embeddings and write to Pinecone
    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(documents, embeddings, index_name="langchain-document-index")
    print("Embeddings written to Pinecone")


if __name__ == "__main__":
    ingest_docs()
