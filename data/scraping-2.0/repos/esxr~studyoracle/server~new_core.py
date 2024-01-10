"""
Interactive interface: We will provide user following interactive options:
1. Add a PDF file
2. Remove a PDF file
3. List all currently added PDF files
4. Query

- There will be a FAISS based vector store.
- There will be an LLM based Agent using that vector store to answer queries.
- There will be a dict mapping each PDF file name(s) to a list of UUIDs for text chunks in that PDF file.
- For each added PDF file:
    - Extract text from PDF file using PdfReader
    - Split text using CharacterTextSplitter. For each text chunk:
        - Generate a UUID
    - add UUIDs to the mapping dict
    - add text chunks (along with UUIDs) to FAISS vector store
- For each query:
    - Use LLM based Agent to answer query
- For each removed PDF file:
    - Pop UUIDs from the mapping dict
    - Remove text chunks (along with UUIDs) from FAISS vector store
"""
import glob
import uuid

from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from io import BytesIO
import boto3
from PyPDF2 import PdfReader

from server.utils import convert_PageObjects_to_Documents


def get_all_pdf_files(directory):
    return glob.glob(f"{directory}/**/*.pdf", recursive=True)


class PDFQA:
    def __init__(self, mapping=None, vectordb=None, embedding_function=None, llm=None, chain=None):
        self.mapping = mapping or {}
        self.embedding_function = embedding_function or OpenAIEmbeddings()
        self.vectordb = vectordb or self.__create_initial_vector_db()
        self.llm = llm or OpenAI()
        self.chain = chain or RetrievalQA.from_chain_type(llm=self.llm,
                                                          chain_type="stuff",
                                                          retriever=self.vectordb.as_retriever())

    def __create_initial_vector_db(self):
        text = """This service generates relevant answers to your queries on the PDF documents you've added.
        When you add a document, it is split into text chunks and each text chunk is converted into an embedding.
        These embeddings are stored in a vector store. When you query, the query is converted into an embedding
        and the vector store is searched for the most similar embeddings. The text chunks corresponding to the
        most similar embeddings are then used to generate an answer for your query (using a large language model)
        """
        # Convert the text into embeddings and create a retriever
        vectordb = FAISS.from_documents([Document(page_content=text)], self.embedding_function)
        return vectordb

    # function to read PDF from an S3 bucket url using boto3
    # the final output should be list[Document]
    def read_pdf_from_s3(self, filename):
        s3 = boto3.resource("s3")
        # TODO: remove hardcoded bucket name
        obj = s3.Object("studyoracle", filename)
        fs = obj.get()["Body"].read()
        reader = PdfReader(BytesIO(fs))

        # convert reader.pages (List[PageObject]) to List[Documents]
        return convert_PageObjects_to_Documents(reader.pages)

    def add_pdf(self, pdf_path):
        docs = self.read_pdf_from_s3(pdf_path)
        # docs = PyPDFLoader(pdf_path).load()
        # Split documents into text chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(docs)
        # Generate UUIDs for each text chunk and add to the mapping
        uuids = [str(uuid.uuid4()) for _ in range(len(texts))]
        # Convert the text into embeddings and add to the vector store
        self.vectordb.add_documents(texts, ids=uuids)
        # Add the UUIDs to the mapping
        self.mapping[pdf_path] = uuids

    def remove_pdf(self, pdf_path):
        # Remove UUIDs from the mapping
        uuids = self.mapping.pop(pdf_path)
        # Remove text chunks (along with UUIDs) from the vector store
        self.vectordb.delete(ids=uuids)

    def list_pdfs(self):
        return list(self.mapping.keys())

    def query(self, query):
        return self.chain.run(query)
