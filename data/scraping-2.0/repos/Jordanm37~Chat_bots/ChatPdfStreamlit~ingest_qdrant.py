from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import os

load_dotenv()

# load the API key from the .env file
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# user_input = input("User: ")
# load the documents from the docs directory
loader = PyPDFDirectoryLoader("docs/")
docs = loader.load()

#split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(docs)

embedding = OpenAIEmbeddings()