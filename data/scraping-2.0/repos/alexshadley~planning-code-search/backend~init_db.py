import uuid
from langchain.embeddings import OpenAIEmbeddings

import os

from backend.load import get_chunks

import dotenv

dotenv.load_dotenv()

import pinecone


def batch(items, size=20):
    batches = []
    while len(items):
        batches.append(items[:size])
        items = items[size:]

    return batches


def setup(relative_path='../corpus/'):
    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="gcp-starter")

    try:
        pinecone.delete_index("planning-code-chunks")
    except pinecone.NotFoundException:
        pass
    pinecone.create_index("planning-code-chunks", dimension=1536)

    embeddings_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

    documents = [relative_path + "san_francisco-ca-2.html" ]
    chunks = get_chunks(documents)

    embeddings = embeddings_model.embed_documents([c.page_content for c in chunks])

    index = pinecone.Index("planning-code-chunks")
    for b in batch(list(zip(embeddings, chunks))):
        print("uploading chunk")
        index.upsert(
            vectors=[
                (str(uuid.uuid4()), e, {"page_content": d.page_content, **d.metadata})
                for (e, d) in b
            ]
        )

if __name__ == "__main__":
    setup()
