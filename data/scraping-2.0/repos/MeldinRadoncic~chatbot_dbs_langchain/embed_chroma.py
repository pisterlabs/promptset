import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
import openai


## State where we want to save the vector database.
PERSIST_DIRECTORY = 'db'

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def load_and_split_documents():
    documents = []
    loader =  TextLoader(os.path.join('./content.txt'), encoding='utf-8')
    ## load and split the documents. load_and_split() function from langchain documentation.

    loaded_docs = loader.load_and_split()
    ## Add the loaded documents to the documents list.
    documents.extend(loaded_docs)

    return documents


def embed_documents(documents):
    embedding = OpenAIEmbeddings()
    vector_db = Chroma.from_documents(documents, embedding, persist_directory=PERSIST_DIRECTORY)
    vector_db.persist()








def main():
    documents = load_and_split_documents()

    embed_documents(documents)

    print("We finished loading and embedding the documents.")

if __name__ == "__main__":
    main()