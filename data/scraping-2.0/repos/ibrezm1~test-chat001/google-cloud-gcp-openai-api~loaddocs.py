from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.document_loaders import UnstructuredFileLoader

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import TensorflowHubEmbeddings

from langchain.vectorstores import DeepLake
from langchain.document_loaders import TextLoader


url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
embed_model = TensorflowHubEmbeddings(model_url=url)

import requests

headers = {
    'Referer': 'https://www.google.com/',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'sec-ch-ua': '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
}

response = requests.get('https://www.cardinalhealth.com/sitemap.xml', headers=headers)

from bs4 import BeautifulSoup

#response = get_response('https://www.cardinalhealth.com/sitemap.xml')
soup = BeautifulSoup(response.content, "xml")
urls = [element.text for element in soup.find_all("loc")]

sites = urls
sites_filtered = [url for url in sites if '/reference/' not in url and '?hl' not in url]


diversitysites = [url for url in sites if 'diversity' in url]

from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = UnstructuredURLLoader(urls=diversitysites[:15], continue_on_failure=False, headers=headers)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 950,
    chunk_overlap  = 90,
    length_function = len,
)

documents = text_splitter.split_documents(documents)

dataset_path = "./deeplakev3"
vectorstore = DeepLake.from_documents(documents, embed_model,dataset_path=dataset_path)

