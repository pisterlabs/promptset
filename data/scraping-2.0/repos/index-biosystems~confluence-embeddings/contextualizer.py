import config
from config import os
import requests
import pinecone

from atlassian import Confluence
from bs4 import BeautifulSoup
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Pinecone

# initialize pinecone
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),  # find at app.pinecone.io
    environment=os.getenv('PINECONE_API_ENV')  # next to api key in console
)

confluence = Confluence(
    url=os.getenv('CONFLUENCE_URL'),
    username=os.getenv('CONFLUENCE_USERNAME'),
    password=os.getenv('CONFLUENCE_API_KEY'))

# Get all pages with the 'embedding' label.

pages = confluence.get_all_pages_by_label('embedding')

# Get all page content.

allPagesContent = []

for page in pages:
    key = page['title']
    r = requests.get(f"{os.getenv('CONFLUENCE_URL')}/wiki/{page['_links']['webui']}", auth=(os.getenv('CONFLUENCE_USERNAME'), os.getenv('CONFLUENCE_API_KEY')))
    soup = BeautifulSoup(r.content, 'html.parser')
    content = soup.find("div", id = 'content').get_text(separator=' ', strip=True)
    allPagesContent.append(content)

# Split up the text and save to the "confluence" index on pinecone.

text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)

allPagesTextSplit = []

for pageContent in allPagesContent:
    allPagesTextSplit.append(text_splitter.split_text(pageContent))

embeddings = OpenAIEmbeddings()

index_name = os.getenv('PINECONE_INDEX_NAME')

for pageTextSplit in allPagesTextSplit:
    Pinecone.from_texts(pageTextSplit, embeddings, index_name=index_name)