import os
import re
import json
import time
from tqdm import tqdm
from urllib import request
from datetime import datetime, timedelta
from dotenv import load_dotenv

import pinecone
from langchain.embeddings import VertexAIEmbeddings
from langchain.vectorstores import Pinecone

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
embeddings = VertexAIEmbeddings()

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment='gcp-starter',
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=100
)

api_key = os.getenv('GNEWS_API_KEY')
today = datetime.today()
str_format = r'%Y-%m-%dT%H:%M:%SZ'
max_retries = 3
timeout = 30

for i in range(30, 0, -1):
    date = today - timedelta(days=i)
    date_str = date.strftime(r'%Y-%m-%d')
    start = date.replace(hour=0, minute=0, second=0).strftime(str_format)
    end = date.replace(hour=23, minute=59, second=59).strftime(str_format)

    url = f"https://gnews.io/api/v4/top-headlines?&lang=en&max=10&from={start}&to={end}&apikey={api_key}"
    with request.urlopen(url) as response:
        data = json.loads(response.read().decode("utf-8"))
        articles = data['articles']
        documents = []
        for news in tqdm(articles, desc=f"Processing Articles on {date_str}"):
            try:
                loader = WebBaseLoader(news['url'])
                doc = loader.load()
                doc[0].page_content = re.sub(r'\s+', ' ', doc[0].page_content)
                documents.extend(doc)
            except:
                print('Skipping news article..\n')

        print('Documents have been loaded! Pushing to vectorDB\n')
        doc_splits = text_splitter.split_documents(documents)

        for i in range(max_retries):
            try:
                docsearch = Pinecone.from_documents(doc_splits, embeddings, index_name='news-data-v2')
                print(f"VectorDB updated with news on {date_str}!\n")
                break
            except:
                print('Retrying embedding request..\n')
                time.sleep(timeout)
