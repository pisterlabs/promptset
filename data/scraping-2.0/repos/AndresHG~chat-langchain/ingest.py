"""Load html from files, clean up, split, ingest into Weaviate."""
import nltk
import glob

from langchain.document_loaders import ReadTheDocsLoader, UnstructuredPDFLoader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

nltk.download("punkt")


def ingest_docs():
    """Get documents from local and load data."""
    documents = []
    for filename in glob.glob("data/PFRPG_SRD_*.pdf"):
        loader = UnstructuredPDFLoader(filename)
        raw_documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
        )
        documents.extend(text_splitter.split_documents(raw_documents))
        print(f"File read - {filename}")

    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large", model_kwargs={"device": "cuda"}
    )
    persist_directory = "db"
    vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)

    vectorstore.persist()
    print("Vector-store saved and ready to use!")


if __name__ == "__main__":
    ingest_docs()
