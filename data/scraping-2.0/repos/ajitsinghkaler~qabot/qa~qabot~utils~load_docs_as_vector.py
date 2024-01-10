from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from io import BytesIO


from qabot.utils.unstructured_pdf import UnstructuredPDFLoader2
from qa.settings import db_directory

def load_docs_as_vector(file):
    file_content  = file.read()
    file_io = BytesIO(file_content)
    extra_data = {"file_name": file.name}
    loader = UnstructuredPDFLoader2(file_io, **extra_data)
    splitter =  RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,)
    pages = loader.load_and_split(splitter)
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=pages, embedding=embeddings, persist_directory=db_directory)
    vectordb.persist()

