import glob
from typing import List

from langchain_core.documents import Document
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector


def chunk_document(doc_path: str) -> List[Document]:
    """Chunk a document into smaller langchain Documents for embedding.

    :param doc_path: path to document
    :type doc_path: str
    :return: List of Document chunks
    :rtype: List[Document]
    """
    loader = PyPDFLoader(doc_path)
    documents = loader.load()

    # split document based on the `\n\n` character, quite unintuitive
    # https://stackoverflow.com/questions/76633836/what-does-langchain-charactertextsplitters-chunk-size-param-even-do
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    return text_splitter.split_documents(documents)


# The connection to the database
CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver="psycopg2",
    host="localhost",
    port="5432",
    database="postgres",
    user="username",
    password="password",
)


def embed_documents(doc_dir: str, add_docs: bool = False):
    # load the document and split it into chunks
    doc_chunks = []
    for doc in glob.glob(f"{doc_dir}/*.pdf"):
        doc_chunks += chunk_document(doc)

    # The embedding function that will be used to store into the database
    embedding_function = SentenceTransformerEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if not add_docs:
        db = PGVector.from_documents(
            doc_chunks,
            connection_string=CONNECTION_STRING,
            collection_name="embeddings",
            embedding=embedding_function,
            pre_delete_collection=True,
        )
        print(f"Created new database with {len(doc_chunks)} embeddings.")
    else:
        db = PGVector(
            connection_string=CONNECTION_STRING,
            collection_name="embeddings",
            embedding=embedding_function,
        )
        res = db.add_documents(doc_chunks)
        print(f"Added {len(res)} embeddings.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--doc_dir", type=str, required=True, help="path to document to embed"
    )
    parser.add_argument("--add", action="store_true", help="add to existing collection")

    args = parser.parse_args()

    embed_documents(args.doc_dir, args.add)
