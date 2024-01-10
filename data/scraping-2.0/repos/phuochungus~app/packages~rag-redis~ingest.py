import os

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Redis
from rag_redis.config import INDEX_NAME, INDEX_SCHEMA, REDIS_URL
from langchain.embeddings import OpenAIEmbeddings


def ingest_documents():
    """
    Ingest PDF to Redis from the data/ directory that
    """
    # Load list of pdfs
    data_path = "data/"
    docs = [os.path.join(data_path, file) for file in os.listdir(data_path)]

    print("Parsing docs", docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=100, add_start_index=True
    )
    chunks = []
    for doc in docs:
        loader = UnstructuredFileLoader(doc, mode="single", strategy="fast")
        chunks.extend(loader.load_and_split(text_splitter))

    print("Chunk 0:", chunks[0])

    print("Done preprocessing. Created", len(chunks), "chunks of the original pdf")

    rds = Redis.from_texts(
        texts=[chunk.page_content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        embedding=OpenAIEmbeddings(),
        index_name=INDEX_NAME,
        redis_url=REDIS_URL,
    )
    rds.write_schema(INDEX_SCHEMA)


if __name__ == "__main__":
    ingest_documents()
