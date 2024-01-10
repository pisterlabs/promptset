### This file is used to generate embeddings for a PDF document and store them in Pinecone

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import pinecone
import openai
import os
import dotenv

##Get and initialise API Keys and environment variables from .env file or user input:
try:
    dotenv.load_dotenv(dotenv.find_dotenv())
    use_env = input("Do you want to use the .env file? (y/n)"+ "\n")

    while use_env.lower() not in ["y", "yes", "n", "no"]:
        print("Not a valid input")
        use_env = input("Do you want to use the .env file? (y/n)\n")

    if use_env.lower() in ["y", "yes"]:
        print("Using .env file.")
        # TODO: get api from .env file and check they are valid
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        print("Using OpenAI API key: " + OPENAI_API_KEY + "\n")
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        print("Using Pinecone API key: " + PINECONE_API_KEY + "\n")
        PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
        print("Using Pinecone environment: " + PINECONE_ENVIRONMENT + "\n")
        PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
        print("Using Pinecone index name: " + PINECONE_INDEX_NAME + "\n")
    else:
        print("No .env file found.")
        print("Please enter your API keys manually.")
        
        OPENAI_API_KEY = input("Enter your OpenAI API key: ")
        PINECONE_API_KEY = input("Enter your Pinecone API key: ")
        PINECONE_ENVIRONMENT = input("Enter your Pinecone environment: ")
        PINECONE_INDEX_NAME = input("Enter your Pinecone index name: ")
except:
    print("No .env file found.")
    # Ask user for OpenAI API key
    while True:
        try:
            OPENAI_API_KEY = input("Enter your OpenAI API key: ")
            openai.api_key = OPENAI_API_KEY
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            print("Using OpenAI API key: " + OPENAI_API_KEY + "\n")
            break
        except:
            print("Invalid API key.")

    # Ask user for Pinecone API key and environment
    while True:
        try:
            PINECONE_API_KEY = input("Enter your Pinecone API key: ")
            PINECONE_ENVIRONMENT = input("Enter your Pinecone environment: ")
            pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
            print("Using Pinecone API key: " + PINECONE_API_KEY)
            print("Using Pinecone environment: " + PINECONE_ENVIRONMENT + "\n")
            break
        except:
            print("Invalid API key or environment.")

    # Ask user for Pinecone index name
    PINECONE_INDEX_NAME = input("Enter your Pinecone index name: ")
    print("Using Pinecone index: " + PINECONE_INDEX_NAME + "\n")



## Initialize OpenAI API key and environment
openai.api_key = OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

## Initialize Pinecone using API key and environment
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index_name = PINECONE_INDEX_NAME

## Set File directory for PDF document
filePath =  os.path.dirname(os.path.realpath(__file__)) + '/docs/' + input("Enter the name of the PDF file you want to create embeddings for: ")

## Load the PDF document
document_loader = UnstructuredPDFLoader(filePath)
data = document_loader.load()
print (f'You currently have {len(data)} document(s) in your data')

## Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0) ## 2000 characters per chunk - change this to your liking
texts = text_splitter.split_documents(data)
print(f'Finished Splitting - Now you have {len(texts)} documents')

## Vectorize the document chunks and turn them into embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
while True:
    try:
        # TODO: this may not be the best way to check if the index exists as it will throw an error if there is a connection issue as well
        docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
        print("Success! Embeddings created and stored in Pinecone index: " + index_name) 
        break
    except:
        print("Index doesn't exist.")
        # Ask user if they want to create a new index
        create_new_index = input("Do you want to create a new index? (y/n)\n").lower()
        while create_new_index != "y" or "yes" or "n" or "no":
            print("not a valid input")
            create_new_index = input("Do you want to create a new index? (y/n)\n").lower()

        if create_new_index in ["y", "yes"]: # User selects "yes"
            try:
                pinecone.delete_index(index_name=index_name)
            except:
                pass
            print("Creating new index...")
            pinecone.create_index(name=index_name, metric="cosine", dimension=1536, pod_type= "p2")
            docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
            print("Success! Embeddings created and stored in Pinecone index: " + index_name)
        else:
            index_name = input("Enter a different Pinecone index name: ") # User selects "no"
            print("Using Pinecone index: " + index_name + "\n")