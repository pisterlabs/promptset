# combine this code into server.py
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import os, getpass

# prompt for API key, only done once at init
os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')

# THIS COSTS MONEY
# It should save the file so it can be reused

if os.path.isdir("data/vecIndex"):
    faiss_index = FAISS.load_local("data/vecIndex", OpenAIEmbeddings())
else:
    option = input("vecIndex does not exist.  Would you like to generate it?  This will use your OpenAPI key. Enter Y to continue: ")
    if option == 'Y':
        loader = PyPDFLoader("data/eng-national-retail-9-1-23.pdf")
        pages = loader.load_and_split()
        faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
        faiss_index.save_local("data/vecIndex")
    else:
        print("Exiting")
        exit()