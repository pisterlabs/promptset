from uuid import uuid4
import os
import pinecone
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv("/Users/syyun/Dropbox (MIT)/lawgpt/.env")

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = "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"
# Initialize Pinecone
PINECONE_API_KEY = "978bf2d9-c8d3-449f-bdfb-5adeb77a6e97"
PINECONE_ENV = "gcp-starter"
from langchain.embeddings.openai import OpenAIEmbeddings

# Initialize Pinecone
index_name = "decision-tree"
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(index_name)


ids = [str(uuid4())]


parent = "Instances where a legislator is assigned to a committee that has been lobbied on by the company in question."
child = "Check if the legislator has had any financial transactions with companies in the same industry as the company in question."

model_name = "text-embedding-ada-002"
embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)

embeds = embed.embed_documents([parent])


metadatas = [
    {"parent": parent, "child": child}
]  # parent is usally the textual description of the data, child is usually the another text description of data or the decision itself.

index.upsert(vectors=zip(ids, embeds, metadatas))
# # delete all before upsert
# index.delete(delete_all=True) # not-working for starter


def (parent, child):
    from uuid import uuid4
    import pinecone

    ids = [str(uuid4())]

    # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_KEY = "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"
    # Initialize Pinecone
    PINECONE_API_KEY = "978bf2d9-c8d3-449f-bdfb-5adeb77a6e97"
    PINECONE_ENV = "gcp-starter"
    from langchain.embeddings.openai import OpenAIEmbeddings

    # Initialize Pinecone
    index_name = "decision-tree"
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    index = pinecone.Index(index_name)

    model_name = "text-embedding-ada-002"
    embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)

    embeds = embed.embed_documents([parent])

    metadatas = [
        {"parent": parent, "child": child}
    ]  # parent is usally the textual description of the data, child is usually the another text description of data or the decision itself.

    index.upsert(vectors=zip(ids, embeds, metadatas))
    return True
