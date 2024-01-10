from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
import os
import tiktoken

from dotenv import load_dotenv

load_dotenv()

INDEX_NAME = "db1"
DOCS_PATH = "elys-docs/docs/"
PINECONE_NAMESPACE = "elys-1"

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV")
)


def ingest_docs() -> None:
    custom_html_tag = ("main", {})

    loader = ReadTheDocsLoader(
        path=DOCS_PATH,
        features="html.parser",
        custom_html_tag=custom_html_tag,
    )
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents")

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=100, length_function=len, add_start_index=True
    )

    documents = text_splitter.split_documents(raw_documents)
    print(f"Split into {len(documents)} chunks")

    print(f"Going to insert {len(documents)} documents to Pinecone")
    embeddings = OpenAIEmbeddings()

    Pinecone(pinecone.Index(INDEX_NAME), embedding=embeddings, text_key="text")

    Pinecone.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=INDEX_NAME,
        namespace=PINECONE_NAMESPACE,
    )


# langchain is done with readthedocs
if __name__ == "__main__":
    ingest_docs()

# tokenizer = tiktoken.get_encoding('cl100k_base')
# def tiktoken_len(text):
#     tokens = tokenizer.encode(
#         text,
#         disallowed_special=()
#     )
#     return len(tokens)

# tiktoken_len("hello I am a chunk of text and using the tiktoken_len function "
#              "we can find the length of this chunk of text in tokens")

# print("deleting namespace")
# vectorstore.delete(
#     delete_all=True, namespace="cosmos-047"
# )


# for doc in documents:
#     old_path = doc.metadata["source"]
#     new_url = old_path.replace("langchain-docs", "https:/")
#     doc.metadata.update({"source": new_url})
