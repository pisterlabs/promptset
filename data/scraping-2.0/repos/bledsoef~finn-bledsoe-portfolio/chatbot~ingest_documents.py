import dotenv
import os
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores.pinecone import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from pathlib import Path

def ingest_documents():
        """
        A script that when run will take the pdf's in your docs folder and add them to the vector database Pinecone.
        """
        env_path = Path('.') / '.env'
        dotenv.load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.environ["open_ai_key"]
        file_path = "chatbot/info.txt"
        try:
            # load in the pdf files in the directory specified by directory_path
            textLoader = TextLoader(f"{file_path}")
            documents = textLoader.load()

            # recursively split the text into vectors to be stored in Pinecone with the size of 1000
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
            split_documents = text_splitter.split_documents(documents)

            # get the OpenAI Embeddings
            embeddings = OpenAIEmbeddings()

            # add the PDF to the gptslackbot Pinecone index.
            pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENVIRONMENT"])
            Pinecone.from_documents(documents=split_documents, embedding=embeddings,
                                    index_name=os.environ["PINECONE_INDEX_NAME"])
        except Exception as e:
            print(f"Error ingesting documents: {str(e)}")

if __name__ == "__main__":
     ingest_documents()