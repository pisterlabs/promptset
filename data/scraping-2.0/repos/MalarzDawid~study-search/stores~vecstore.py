from pathlib import Path

VECTOR_DIR = Path("data/cache/vectorstore")


def connect_to_vector_db(index_name: str, embedding_engine):
    """Connect to local FAISS vectorstore"""
    from langchain.vectorstores import FAISS

    vector_db = FAISS.load_local(VECTOR_DIR / index_name, embedding_engine, index_name)

    return vector_db


def get_openai_embedding_engine(model: str = "text-embedding-ada-002", **kwargs):
    """Load OpenAI Embeddings with specific model"""
    from langchain.embeddings import OpenAIEmbeddings

    embedding_engine = OpenAIEmbeddings(model=model, **kwargs)
    return embedding_engine


def create_vector_db(index_name: str, embedding_engine, documents: list, metadatas: list, save: bool = False):
    from langchain import FAISS

    index = FAISS.from_texts(
        texts=documents, embedding=embedding_engine, metadatas=metadatas
    )
    if save:
        index.save_local(VECTOR_DIR / index_name, index_name=index_name)
    return index


def prep_documents_for_vecstore(documents: list):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100, allowed_special="all"
    )
    texts, metadatas = [], []
    for document in documents:
        text, metadata = document["text"], document["metadata"]
        doc_texts = text_splitter.split_text(text)
        doc_metadatas = [metadata] * len(doc_texts)
        texts += doc_texts
        metadatas += doc_metadatas
    return texts, metadatas
