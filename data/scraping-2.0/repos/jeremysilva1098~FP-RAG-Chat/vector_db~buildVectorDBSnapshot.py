from langchain.document_loaders import RecursiveUrlLoader
from qdrant_client import QdrantClient, models
import requests
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import os
import dotenv
import random

# load dot env from parent directory
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
dotenv.load_dotenv(dotenv_path)

### load url content and chunk it ###
urls = [
    "https://docs.freeplay.ai/docs", # all the docs content
    "https://freeplay.ai/blog" # all the blog content
    ]

all_docs = []

for url in urls:
    loader = RecursiveUrlLoader(url=url, max_depth=10,
                                extractor=lambda x: BeautifulSoup(x, "html.parser").text)
    docs = loader.load()
    all_docs.extend(docs)
    # view 1 random doc
    print(random.choice(docs))
    print("\n\n")

# split all docs into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(all_docs)
# view 1 random chunk
print(random.choice(chunks))
print("\n\n")
print("Number of chunks: ", len(chunks))

### add the chunks to the vector db ###
# create the vector db client
dbClient = QdrantClient(host='localhost', port=6333)
collection = 'freeplay_content'
localUrl = "http://localhost:6333"
# create the collection
dbClient.create_collection(collection_name=collection,
                           vectors_config=models.VectorParams(
                               size=1536, distance=models.Distance.COSINE)
                        )


# configure openai
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text):
    try:
        response = openai.Embedding.create(input=text,
                                           model="text-embedding-ada-002")
        return response["data"][0]["embedding"]
    except:
        try:
            response = openai.Embedding.create(input=text,
                                               model="text-embedding-ada-002")
            return response["data"][0]["embedding"]
        except:
            print("Back to back Error with text: ", text)
            return None


dbClient.upload_records(
    collection_name=collection,
    records=[
        models.Record(
            id=idx,
            vector=get_embedding(doc.page_content),
            payload={
                "source": doc.metadata['source'],
                "title": doc.metadata['title'],
                "description": doc.metadata['description'],
                "text": doc.page_content}
        ) for idx, doc in enumerate(chunks)
    ]
)

### snapshot the db and write to disk ###
res = requests.post(localUrl + "/snapshots")
print(res.json())
print(res.text)

snapshots = dbClient.list_full_snapshots()
print(snapshots)

# download the latest snapshot
res = requests.get(localUrl + "/snapshots/" + snapshots[0].name)
# write snapshot to disk
with open("latest.snapshot", "wb") as f:
    f.write(res.content)