
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

VECTOR_STORE_PATH = "vector_store.faiss"

def save_to_vector_store(file_content):
    """Saves the file content to the vector store using LangChain's methods.

    Args:
    - file_content (str): The content of the file to be saved.

    Returns:
    - str: A unique identifier for the saved content. (In this mockup, we're not really returning a unique ID,
           but in a real-world scenario, you'd return some identifier to later retrieve the exact content.)
    """
    # Load the file content
    documents = [file_content]

    # Split the content
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Embed the content
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)

    # Save the embeddings (for the sake of this example, we're assuming the FAISS vector store is saved to a file)
    db.save(VECTOR_STORE_PATH)

    return "Saved"


def get_from_vector_store(query):
    """Retrieves relevant documents from the vector store using LangChain's methods.

    Args:
    - query (str): The query to search for in the vector store.

    Returns:
    - list: The relevant documents retrieved from the vector store.
    """
    # Load the vector store
    db = FAISS.load(VECTOR_STORE_PATH)

    # Retrieve relevant documents
    retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5})
    docs = retriever.get_relevant_documents(query)

    return docs
