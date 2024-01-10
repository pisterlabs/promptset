import torch
from langchain_core.documents.base import Document
from langchain.text_splitter import HTMLHeaderTextSplitter
from langchain.document_loaders import AsyncHtmlLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from vectorstore_argparser import VectorStoreArgParser
from typing import Dict, List

DB_FAISS_PATH = "vectorstores/db_faiss"
SEARCH_URL = "https://genshin-impact.fandom.com/wiki/Furina/Lore"
HEADERS_TO_SPLIT_ON = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]
LAST_INDEX_OF_LORE = 24
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def isLoreInfo(metadata: Dict[str, str]):
    return len(metadata) != 0


def refined_document(documents: List[Document]):
    # Replace LAST_INDEX_OF_LORE with the actual index you want to iterate until
    last_index = LAST_INDEX_OF_LORE if LAST_INDEX_OF_LORE < len(documents) else len(documents)

    filtered_docs = filter(lambda doc: isLoreInfo(doc.metadata), documents[:last_index])
    return list(filtered_docs)


def create_vector_store(args: VectorStoreArgParser):
    loader = AsyncHtmlLoader(args.search_url)
    documents = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs= {'device': DEVICE})

    text_splitter = HTMLHeaderTextSplitter(HEADERS_TO_SPLIT_ON)
    texts = refined_document(
        documents = text_splitter.split_text(text=documents[0].page_content)
    )

    if args.save_vector_store:
        try:
            db = FAISS.from_documents(texts, embeddings)
            db.save_local(DB_FAISS_PATH)
            print("Success save vector store")
        except Exception as e:
            print(f"An error occurred: {e}")

    print(f"Length of texts: {len(texts)}")

if __name__ == "__main__":
    arg_parser = VectorStoreArgParser()
    arguments = arg_parser.parse_args()
    create_vector_store(arguments)
