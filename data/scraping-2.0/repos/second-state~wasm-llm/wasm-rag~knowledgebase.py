import os

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

CHROMA_DB_DIRECTORY = "db"
DOCUMENT_SOURCE_DIRECTORY = "source_documents"
TARGET_SOURCE_CHUNKS = 3
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
HIDE_SOURCE_DOCUMENTS = False


class MyKnowledgeBase:
    def __init__(self, pdf_source_folder_path: str) -> None:
        """
        Loads pdf and creates a Knowledge base using the Chroma
        vector DB.
        Args:
            pdf_source_folder_path (str): The source folder containing
            all the pdf documents
        """
        self.pdf_source_folder_path = pdf_source_folder_path

    def load_pdfs(self):
        """Load all the pdfs inside the directory"""
        loader = DirectoryLoader(self.pdf_source_folder_path, show_progress=True)
        loaded_pdfs = loader.load()
        return loaded_pdfs

    def split_documents(
        self,
        loaded_docs,
    ):
        """Split the documents into chunks and return the chunked docs"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunked_docs = splitter.split_documents(loaded_docs)
        return chunked_docs

    def convert_document_to_embeddings(self, chunked_docs, embedder):
        """Convert the chunked docs to embeddings and add that to vector db"""

        # instantiate the Chroma db client
        vector_db = Chroma(
            persist_directory=CHROMA_DB_DIRECTORY,
            embedding_function=embedder,
        )

        # inject the chunks and save all inside the db directory
        vector_db.add_documents(chunked_docs)
        vector_db.persist()

        # finally return the vector db client object
        return vector_db

    def retriever_from_persistant_vector_db(self, embedder):
        """Get a retriever object from the persistant vector db"""
        if not os.path.isdir(CHROMA_DB_DIRECTORY):
            raise NotADirectoryError("Please load your vector database first.")

        vector_db = Chroma(
            persist_directory=CHROMA_DB_DIRECTORY,
            embedding_function=embedder,
        )

        return vector_db.as_retriever(search_kwargs={"k": TARGET_SOURCE_CHUNKS})

    def initiate_document_injection_pipeline(self, embedder):
        print("[INFO] PDF loading and chunking ...")
        loaded_pdfs = self.load_pdfs()
        chunked_documents = self.split_documents(loaded_docs=loaded_pdfs)

        print(
            "[INFO] Converting documents to embeddings and injecting into vector db ..."
        )
        vector_db = self.convert_document_to_embeddings(
            chunked_docs=chunked_documents, embedder=embedder
        )
