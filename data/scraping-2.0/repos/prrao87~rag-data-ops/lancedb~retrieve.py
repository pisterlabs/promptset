import os
import shutil
from pathlib import Path

import lancedb
from lancedb.pydantic import LanceModel, Vector, pydantic_to_schema
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import LanceDB


# LanceDB pydantic schema
class Content(LanceModel):
    text: str
    vector: Vector(384)


def get_files() -> list[str]:
    # Get a list of files from the data directory
    data_dir = Path("../data")
    txt_files = list(data_dir.glob("*.txt"))
    # Return string of paths or else lancedb/pydantic will complain
    txt_files = [str(f) for f in txt_files]
    return txt_files


def get_docs(txt_files: list[str]):
    loaders = [TextLoader(f) for f in txt_files]
    docs = [loader.load() for loader in loaders]
    return docs


def create_lance_table(table_name: str) -> lancedb.table.LanceTable:
    try:
        # Create empty table if it does not exist
        tbl = db.create_table(table_name, schema=pydantic_to_schema(Content), mode="overwrite")
    except OSError:
        # If table exists, open it
        tbl = db.open_table(table_name, mode="append")
    return tbl


async def search_lancedb(query: str, retriever: LanceDB) -> list[Content]:
    "Perform async retrieval from LanceDB"
    search_result = await retriever.asimilarity_search(query, k=5)
    if len(search_result) > 0:
        print(search_result[0].page_content)
    else:
        print("Failed to find similar result")
    return search_result


def main() -> None:
    txt_files = get_files()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
    )
    tbl = create_lance_table("countries")
    docs = get_docs(txt_files)
    chunked_docs = []
    for doc in docs:
        chunked_docs.extend(text_splitter.split_documents(doc))
    # Ingest docs in append mode
    retriever = LanceDB.from_documents(chunked_docs, embeddings, connection=tbl)
    return retriever


if __name__ == "__main__":
    DB_NAME = "./db"
    TABLE = "countries"
    if os.path.exists(DB_NAME):
        # Clear DB if it exists
        shutil.rmtree(DB_NAME)

    db = lancedb.connect(DB_NAME)
    retriever = main()
    print("Finished loading documents to LanceDB")

    query = "Is Tonga a monarchy or a democracy"
    docsearch = retriever.as_retriever(
        search_kwargs={"k": 3, "threshold": 0.8, "return_vector": False}
    )

    search_result = docsearch.get_relevant_documents(query)

    if len(search_result) > 0:
        print(f"Found {len(search_result)} relevant results")
        print([r.page_content for r in search_result])
    else:
        print("Failed to find relevant result")