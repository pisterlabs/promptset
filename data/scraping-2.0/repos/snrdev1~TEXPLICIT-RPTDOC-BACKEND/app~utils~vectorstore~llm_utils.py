from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config import Config


def load_llm():
    """
    The function `load_llm` loads a language model for chat-based applications.

    Returns:
      The function `load_llm()` returns an instance of the `ChatOpenAI` class with a temperature of 0
    and the GPT-3.5 model.
    """
    llm = ChatOpenAI(temperature=0, model=Config.FAST_LLM_MODEL)
    return llm


def get_embeddings():
    """
    The function `get_embeddings` returns a HuggingFaceEmbeddings object initialized with the
    "sentence-transformers/all-MiniLM-L6-v2" model and running on the CPU.

    Returns:
      the HuggingFaceEmbeddings object.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    return embeddings


def get_text_splitter():
    """
    The function `get_text_splitter` returns an instance of the `RecursiveCharacterTextSplitter` class
    with specific parameters.

    Returns:
      an instance of the `RecursiveCharacterTextSplitter` class with the specified parameters
    `chunk_size=4000` and `chunk_overlap=100`.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=100)

    return text_splitter


def split_text(docs):
    """
    The function `split_text` takes a list of documents as input and splits each document into smaller
    parts using a text splitter, returning the splits.

    Args:
      docs: The "docs" parameter is a list of documents that you want to split. Each document can be a
    string containing text.

    Returns:
      the splits of the input documents.
    """
    text_splitter = get_text_splitter()
    splits = text_splitter.split_documents(docs)

    return splits
