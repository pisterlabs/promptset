import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

open_api_key = "<open-ai key>"

os.environ['OPENAI_API_KEY'] = open_api_key

loader = PyPDFLoader("example_data/Chemistry2e-WEB.pdf")
pages = loader.load_and_split()

print(len(pages))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
)

texts = text_splitter.split_documents(pages)

print(len(texts))

embedding = OpenAIEmbeddings(openai_api_key=open_api_key)

print("Embedding texts...")

pinecone.init(
    api_key="<pincone-key>",
    environment="<pincone env>",
)

index_name = "chemistry-2e"

docsearch = Pinecone.from_texts(
    [t.page_content for t in texts],
    embedding=embedding, index_name=index_name)

print("Done embedding texts.")
