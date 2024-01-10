import os
from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, FAISS

from PyPDF2 import PdfReader
import camelot

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class VectorDBConstructor:

    def __init__(self, data_source, pdf_path, embedding, db):
        self.data_source = data_source
        self.pdf_path = pdf_path
        self.embedding = embedding
        self.db = db

    def create_pdf_list(self, pdf_path):

        pdf_docs = []

        print(f"\nPreparing pdf list from {self.pdf_path}")
        for path in os.listdir(self.pdf_path):
            if path.endswith(".pdf"):
                pdf_docs.append(self.pdf_path+"/"+path)

        return pdf_docs

    def read_pdf(self, pdf_docs, data_source):

        if os.path.isfile(f"./backend/{self.data_source}_text.txt"):
            print("Using pre-generated text file...")
            text = f"./backend/{self.data_source}_text.txt"
            return text
        else:
            text=""
            print(f"\nReading pdf documents from {self.pdf_path}...")
            i = 1
            for pdf in pdf_docs:
                print(f"Reading {pdf} ({i}/{len(pdf_docs)})")
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    page.extract_text()
                    text += page.extract_text()
                i += 1

            print("\nWriting text to file...")

            with open(f"./backend/{self.data_source}_text.txt", "a") as f:
                f.write(text)

            return text

    def chunk_pdf(self, text):
        print("\nChunking text...")

        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 1000,  # Need to variablise
            chunk_overlap = 200, # Need to variablise
            length_function = len
        )

        chunks = text_splitter.split_text(text)

        return chunks

    def create_vectorDB(self, chunks, data_source, embedding, db):
        print("Creating search vectorDB...")

        if self.embedding == "openai": 
            embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

        if self.embedding == "hugging":
            embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

        if self.db == "chroma": 
            persist_directory = "./backend/db/"+self.data_source+"_chroma_"+self.embedding
            print("Instantiating persistence ChromaDB instance...")
            vectordb = Chroma.from_texts(chunks, embeddings, metadata=[{"source": str(i)} for i in range(len(chunks))], persist_directory=persist_directory)
            print("--- FIN ---")

        if self.db == "faiss": 
            print("Instantiating persistence FAISS instance...")
            vectordb = FAISS.from_texts(chunks, embeddings)
            vectordb.save_local("./backend/db/"+self.data_source+"_faiss_"+self.embedding)
            print("--- FIN ---")