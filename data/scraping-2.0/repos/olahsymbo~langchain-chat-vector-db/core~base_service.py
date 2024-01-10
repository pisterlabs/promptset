import os
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_KEY")

os.environ["OPENAI_API_KEY"] = API_KEY


class ChatService:

    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 10
    embeddings = OpenAIEmbeddings()

    def __init__(self, saved_file_path, persist_directory):
        self.saved_file_path = saved_file_path
        self.loaders = None
        self.persist_directory = persist_directory

    def create_vector_index(self):
        documents = self.loaders.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.CHUNK_SIZE, chunk_overlap=self.CHUNK_OVERLAP)
        texts = text_splitter.split_documents(documents)
        vectordb = Chroma.from_documents(documents=texts,
                                         embedding=self.embeddings,
                                         persist_directory=self.persist_directory)
        vectordb.persist()

    def query_document(self, prompt):

        vectordb_cont = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        retriever = vectordb_cont.as_retriever(search_kwargs={"k": 3})
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        if prompt:
            query = f"###Prompt {prompt}"
            llm_response = qa(query)
            return llm_response["result"]
        else:
            return []
