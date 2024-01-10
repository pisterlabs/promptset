import os

from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT"],
)


def ingest_docs() -> None:
    loader = ReadTheDocsLoader(
        path="data/langchain.readthedocs.io/en/latest", encoding="utf8"
    )
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Split {len(documents)} documents")
    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("data", "https:/")
        new_url = new_url.replace("\\", "/")
        doc.metadata.update({"source": new_url})
    print(f"Inserting {len(documents)} documents to Pinecone...")
    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=os.environ["PINECONE_INDEX_NAME"],
    )
    print("******* Added documents to Pinecone *******")


if __name__ == "__main__":
    ingest_docs()
