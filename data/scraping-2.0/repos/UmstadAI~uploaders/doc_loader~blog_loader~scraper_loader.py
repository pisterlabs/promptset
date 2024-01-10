import requests
import glob
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "OPENAI_API_KEY")
import pinecone
import time

from uuid import uuid4
from bs4 import BeautifulSoup

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv(), override=True)  # read local .env file

pinecone_api_key = os.getenv("PINECONE_API_KEY") or "YOUR_API_KEY"
pinecone_env = os.getenv("PINECONE_ENVIRONMENT") or "YOUR_ENV"

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/119.0"
}


def get_and_load():
    def get_blog_links(url):
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")

        links = soup.find_all("a", href=True)
        blog_links = [
            link["href"]
            for link in links
            if "blog/" in link["href"]
            and "page" not in link["href"]
            and "?by" not in link["href"]
            and "?cat" not in link["href"]
            and "?date" not in link["href"]
        ]
        return blog_links

    blogs = []
    for page in range(1, 18):
        main_blog_url = f"https://minaprotocol.com/blog/page/{page}"
        blog_links = get_blog_links(main_blog_url)

        blogs.extend(blog_links)

    blogs = list(set(blogs))

    docs = []
    for batch in range(0, len(blogs) // 5, 10):
        print("Batch", batch)
        loader = WebBaseLoader(blogs[batch : batch + 10])
        data = loader.load()
        docs.extend(data)

    return docs


docs = get_and_load()


def upsert(docs):
    # SPLITTING
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=72,
    )

    # IMPORTANT VARIABLE
    splitted_docs = text_splitter.split_documents(docs)

    # EMBEDDING
    model_name = "text-embedding-ada-002"
    texts = [c.page_content for c in splitted_docs]
    metadatas = [c.metadata for c in splitted_docs]

    print("Created", len(texts), "texts")

    chunks = [
        texts[i : (i + 1000) if (i + 1000) < len(texts) else len(texts)]
        for i in range(0, len(texts), 1000)
    ]
    embeds = []

    print("Have", len(chunks), "chunks")
    print("Last chunk has", len(chunks[-1]), "texts")

    for chunk, i in zip(chunks, range(len(chunks))):
        print("Chunk", i, "of", len(chunk))
        new_embeddings = client.embeddings.create(input=chunk, model=model_name)
        new_embeds = [emb.embedding for emb in new_embeddings.data]

        embeds.extend(new_embeds)
        time.sleep(3)

    # PINECONE STORE
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

    index_name = "zkappumstad"
    index = pinecone.Index(index_name)
    print(index.describe_index_stats())

    ids = [str(uuid4()) for _ in range(len(splitted_docs))]

    vector_type = os.getenv("DOCS_VECTOR_TYPE") or "DOCS_VECTOR_TYPE"

    vectors = [
        (
            ids[i],
            embeds[i],
            {
                "text": texts[i],
                "title": metadatas[i]["title"],
                "vector_type": vector_type,
            },
        )
        for i in range(len(splitted_docs))
    ]

    for i in range(0, len(vectors), 100):
        print("Upsertin Batch", i)
        batch = vectors[i : i + 100]
        index.upsert(batch)

    time.sleep(10)
    print(index.describe_index_stats())


upsert(docs)
print("Scraper Loader completed!")
