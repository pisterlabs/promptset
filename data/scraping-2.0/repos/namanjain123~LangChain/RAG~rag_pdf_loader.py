#Setup The Azure Open AI with Langchain
import openai
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import AzureOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
def Load_pdf():
    OPENAI_API_KEY = "PLEASE_ENTER_YOUR_OWNED_AOAI_SERVICE_KEY"
    OPENAI_DEPLOYMENT_NAME = "PLEASE_ENTER_YOUR_OWNED_AOAI_TEXT_MODEL_NAME"
    OPENAI_EMBEDDING_MODEL_NAME = "PLEASE_ENTER_YOUR_OWNED_AOAI_EMBEDDING_MODEL_NAME"
    MODEL_NAME = "text-davinci-003"
    openai.api_type = "azure"
    openai.api_base = "https://PLESAE_ENTER_YOUR_OWNED_AOAI_RESOURCE_NAME.openai.azure.com/"
    openai.api_version = "2022-12-01"
    openai.api_key = "PLEASE_ENTER_YOUR_OWNED_A"

    # Load PDF
    loaders = [
        PyPDFLoader("A:\LangChain\RAG\pdfs\semantic-kernel.pdf")
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    #Splliting
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap = 150
    )
    splits = text_splitter.split_documents(docs)
    #Embeddings
    embedding = OpenAIEmbeddings()

    #Connecting Quadrant
    url = "0.0.0.0:6500"
    qdrant = Qdrant.from_documents(
        docs, 
        embedding, 
        url, 
        prefer_grpc=True, 
        collection_name="hr_docs",
    )
