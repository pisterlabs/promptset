import os
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.neo4j_vector import Neo4jVector
from streamlit.logger import get_logger

from chains import (
    load_embedding_model,
    load_llm,
)

load_dotenv(".env")

#Neo4j config
url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
os.environ["NEO4J_URL"] = url

ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")

logger = get_logger(__name__)
embeddings, dimension = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
)


def main():
    loader = PyPDFDirectoryLoader('./pdfs')
    documents = loader.load_and_split()
    chunks, file_names = split_documents(documents)
    for idx, chunk in enumerate(chunks):
        vectorstore = upload_chunk_to_neo4j(chunk, file_names[idx])


def split_documents(documents):
    chunks = []
    file_names = []
    for doc in documents:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks.append(text_splitter.split_text(text=doc.page_content))
        file_name = doc.metadata.get("source").split("pdfs/")[1]
        file_names.append(file_name)
    return chunks, file_names


def upload_chunk_to_neo4j(chunk, file_name):
    vectorstore = Neo4jVector.from_texts(
        chunk,
        url=url,
        username=username,
        password=password,
        embedding=embeddings,
        metadatas=[{"file_name": file_name}],
        index_name="pdf",
        node_label="PdfChunk",
        pre_delete_collection=False,  # Delete existing PDF data
    )
    return vectorstore


if __name__ == "__main__":
    main()
