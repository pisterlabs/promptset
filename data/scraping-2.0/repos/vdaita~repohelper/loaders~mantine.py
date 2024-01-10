from supabase import create_client
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from dotenv import load_dotenv
import requests
import os
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
import re
from trafilatura import fetch_url, extract

load_dotenv()

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SECRET_KEY")

supabase = create_client(supabase_url, supabase_key)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=0)

embeddings_model = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

urls = []
docs = []

with open("sidebar.html", "r") as f:
    content = f.read()
    pattern = r'href="([^"]+)"'
    matches = re.findall(pattern, content)
    for match in matches:
        url = "https://mantine.dev" + match
        urls.append("https://mantine.dev" + match)
        downloaded = fetch_url(url)
        if not(downloaded):
            print("Failed load of: " + url)
            continue
        result = extract(downloaded, include_links=True, include_tables=True)
        print(result)
        docs.append(Document(page_content=result, metadata={"title": url,"source": url}))
        



print(docs)
docs = text_splitter.split_documents(docs)

docs_strings = []
for doc in docs:
    docs_strings.append(doc.page_content)
embeddings_vector = embeddings_model.embed_documents(docs_strings)

data = []

for i in range(len(docs)):
    data.append({
        "embedding": embeddings_vector[i],
        "metadata": docs[i].metadata,
        "content": docs[i].page_content,
        "repo": "mantine"
    })


for i in range(0, len(data), 10):
    supabase.table("documents").insert(data[i:i+10]).execute()
