import os

import pinecone
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

from constants import CONTRACT_DIR, PINECONE_INDEX

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

contracts_directory = CONTRACT_DIR

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)

def ingest_docs() -> None:
    loader = DirectoryLoader(
        "../",
        glob="**/*.sol",
        use_multithreading=True,
        show_progress=True,
        loader_cls=TextLoader,
    )
    raw_documents = loader.load()

    print(f"Found {len(raw_documents)} contract files")

    index = pinecone.Index(PINECONE_INDEX)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    sol_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.SOL, chunk_size=1000, chunk_overlap=100
    )

    documents = sol_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} chunks")

    print(f"Going to insert {len(documents)} to Pinecone")
    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(documents, embeddings, index_name="smart-contract-index")

    print("****** Added to Pinecone vectorstore vectors ******")


if __name__ == "__main__":
    ingest_docs()
