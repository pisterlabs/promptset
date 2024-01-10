import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.vectorstores import Pinecone
import pinecone
import openai



# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

## Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment = os.getenv("PINECONE_ENV")) 


def load_and_split_documents(doc_path):
    loader = PyPDFLoader(os.path.join(doc_path))
    ## load and split the documents. load_and_split() function from langchain documentation.
    documents = loader.load_and_split()

    return documents


def embed_documents(documents):
    embedding = OpenAIEmbeddings()
    Pinecone.from_documents(documents, embedding, index_name = "pinecone-chatbot")
    




def main():
    documents_path = ["intercom_1.pdf", "intercom_2.pdf"]

    for doc_path in documents_path:
        documents = load_and_split_documents(doc_path)
        embed_documents(documents)

    print("We finished loading and embedding the documents.")


if __name__ == "__main__":
    main()
