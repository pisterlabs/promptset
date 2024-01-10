from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders.pdf import PDFMinerLoader
from langchain.document_loaders.csv_loader import CSVLoader
import time
import logging


class VecBase:
    def __init__(self, embedding_model_name, chunk_size=1000, chunk_overlap=200):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            is_separator_regex=False,
            length_function=len,
        )

        embeddings_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )

        csv_loader = DirectoryLoader("tinkoff-terms", glob="**/*.csv", loader_cls=CSVLoader)
        csv_chunks = text_splitter.split_documents(csv_loader.load())
        pdf_loader = DirectoryLoader("tinkoff-terms", glob="**/*.pdf", loader_cls=PDFMinerLoader)
        pdf_chunks = text_splitter.split_documents(pdf_loader.load())

        start = time.time()
        self.db = Chroma.from_documents(pdf_chunks + csv_chunks, embeddings_model)
        logging.info(f"time for db creation: {round(time.time() - start)} seconds")

    def similarity_search(self, query, k=4):
        res = []
        for doc in self.db.similarity_search(query, k):
            res.append(doc.page_content)

        return "\n".join(res)
