import openai
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

loader = UnstructuredPDFLoader("docs/QM485-1122 Business Pack Insurance Policy.pdf")
documents = loader.load()

# split it into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=0)

docs = text_splitter.split_documents(documents)

# load it into Chroma
vectorstore = Chroma.from_documents(docs,
                                    OpenAIEmbeddings(),
                                    persist_directory="./vectorstore", )
