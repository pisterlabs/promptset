from pathlib import Path
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceEndpoint
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import DeepLake
from dotenv import load_dotenv

