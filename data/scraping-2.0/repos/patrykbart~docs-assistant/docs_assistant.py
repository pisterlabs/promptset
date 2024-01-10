from os import path, listdir
from pypdf import PdfReader
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

class DocsAssistant:

    def __init__(self, llm_name="google/flan-t5-xxl", embed_model_name="hkunlp/instructor-xl", chunk_size=512, chunk_overlap=64):
        self.llm_name = llm_name
        self.embed_model_name = embed_model_name

        self.splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)

        self.vector_store = None
        self.conv_chain = None

    def embed_pdfs(self, pdfs_root_path="./pdf_pages"):
        raw_text = self.load_pdfs(pdfs_root_path)
        chunks = self.chunk_text(raw_text)
        self.generate_vector_store(chunks)
        self.generate_conversational_chain()

    def load_pdfs(self, pdfs_root_path):
        text = ""
        pdf_files = [path.join(pdfs_root_path, f) for f in listdir(pdfs_root_path) if f.endswith(".pdf")]

        for pdf_path in pdf_files:
            pdf = PdfReader(pdf_path)

            for page in pdf.pages:
                text += page.extract_text()

        return text
    
    def chunk_text(self, text):
        chunks = self.splitter.split_text(text)
        return chunks
    
    def generate_vector_store(self, chunks):
        embeddings = HuggingFaceInstructEmbeddings(model_name=self.embed_model_name)
        self.vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
        
    def generate_conversational_chain(self):
        llm = HuggingFaceHub(repo_id=self.llm_name, model_kwargs={"temperature": 0.5, "max_length": 256})
        self.conv_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=self.vector_store.as_retriever())

    def answer_question(self, query):
        response = self.conv_chain({"question": query, "chat_history": ""})["answer"]
        return response

