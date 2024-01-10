import os
import openai
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
#read .env file
from dotenv import load_dotenv
load_dotenv()

# os.environ["OPENAI_API_KEY"] #= os.getenv('OPENAI_API_KEY')
# os.environ["AZURE_OPENAI_ENDPOINT"] #= os.getenv('AZURE_OPENAI_ENDPOINT')

def OpenAIembeddings():
    open_ai_embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        openai_api_version="2023-05-15",
        chunk_size=10,
    )
    return open_ai_embeddings


def qdrant_load_by_chunks(documents, embedding_model, destination_folder, chunk_size=1500, overlap=100):
    """
    Method to embed and store documents in Qdrant database.
    Splits the documents and waits for 1 minute to continue embedding when a RateLimitError is raised.
    """
    # Splitting the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, add_start_index=True)
    chunked_documents = text_splitter.split_documents(documents)

    # Initialize Qdrant database
    qdrant = Qdrant.from_documents(documents=chunked_documents, embedding=embedding_model, path=destination_folder, collection_name="financebench")

    for document in chunked_documents:
        try:
            qdrant.add_documents(document)
        except Exception as e:
            # Wait for 1 minute before retrying
            time.sleep(60)
            qdrant.add_documents(document)
        # Wait for 1 second to avoid overloading the Qdrant server
        time.sleep(1)

    return qdrant


def llm():
    return AzureChatOpenAI(model_name="gtp35turbo-latest")


