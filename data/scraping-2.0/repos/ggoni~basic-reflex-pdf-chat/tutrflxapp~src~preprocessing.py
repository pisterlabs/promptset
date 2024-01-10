from functools import partial
import glob
import os
import openai
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from typing import List
from langchain.llms import HuggingFaceHub
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI

load_dotenv()

openai.openai_api_key = os.getenv("OPENAI_API_KEY")


class PdfVectorizer():
    def __init__(self, pdf_docs: List[str]):
        """
        Initialize the PdfVectorizer class.

        Args:

        pdf_docs: Pdfs list

        """
        self.pdf_docs = pdf_docs
        self.vector_store = None
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len)
        self.qa = None

    def get_pdf_text(self) -> str:
        """
        Extract text from PDF documents.

        Args:
            pdf_docs (List[str]): List of paths to PDF documents.

        Returns:
            str: The extracted text from the PDF documents.
        """
        raw_text = ""
        for pdf in self.pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                raw_text += page.extract_text()
        return raw_text

    def get_text_chunks(self) -> List[str]:
        """
        Split the raw text into smaller chunks.

        Args:
            raw_text (str): The raw text to be split.

        Returns:
            List[str]: The list of text chunks.
        """
        raw_text = self.get_pdf_text()
        text_chunks = self.text_splitter.split_text(raw_text)
        return text_chunks

    def get_vector_store(self) -> FAISS:
        """
        Create a vector store from the PDF documents.

        Args:
            pdf_docs (List[str]): List of paths to PDF documents.

        Returns:
            FAISS: The created vector store.
        """
        text_chunks = self.get_text_chunks()
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(
                texts=text_chunks, embedding=self.embeddings)
        return self.vector_store


# pdf_docs = glob.glob("../../.web/public/*.pdf")
# pdf_file_names = [os.path.basename(pdf_doc) for pdf_doc in pdf_docs]
# print(pdf_file_names)

# PdfVectorizer = PdfVectorizer(pdf_docs)
# print(PdfVectorizer.get_pdf_text())

# print("Ahora los chunks...")

# print(PdfVectorizer.get_text_chunks())

# vectorizador = PdfVectorizer.get_vector_store()
# print(vectorizador.__sizeof__)
