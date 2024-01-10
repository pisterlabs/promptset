
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import TensorflowHubEmbeddings

from langchain.vectorstores import DeepLake
from langchain.document_loaders import TextLoader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.document_loaders import UnstructuredFileLoader

dataset_path = "./deeplakev3"

url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
embed_model = TensorflowHubEmbeddings(model_url=url)

readdb = DeepLake(dataset_path=dataset_path, read_only=True, embedding=embed_model)
query = "What is diversity"
docs = readdb.similarity_search(query)
print(docs[0].page_content)