import asyncio
import aiohttp
import logging
import pandas as pd
from uuid import uuid4
from tqdm import tqdm
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from config import OPENAI_API_KEY

# Constants
MODEL_NAME = 'text-embedding-ada-002'
EMBED_DIM = 1536
BATCH_LIMIT = 100
INDEX_NAME = 'threat-data'

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize aiohttp session
session = aiohttp.ClientSession()

async def generate_embeddings(texts):
    """
    Asynchronously generate embeddings for a list of texts.
    """
    embed = OpenAIEmbeddings(model=MODEL_NAME, openai_api_key=OPENAI_API_KEY)
    return embed.embed_documents(texts)

async def upsert(vectors):
    """
    Asynchronously upsert vectors to Pinecone.
    """
    if INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(name=INDEX_NAME, metric='cosine', dimension=EMBED_DIM)
    index = pinecone.Index(INDEX_NAME)
    index.upsert(vectors=vectors)

async def process_batch(batch):
    """
    Asynchronously process a batch of data.
    Generate embeddings for the texts in the batch and upsert them to Pinecone.
    """
    texts, metadatas = batch
    ids = [str(uuid4()) for _ in range(len(texts))]
    embeds = await generate_embeddings(texts)
    vectors = list(zip(ids, embeds, metadatas))
    await upsert(vectors)

async def send_embeddings_to_pinecone_async(data_directory):
    """
    Asynchronously process the provided data in batches.
    Generate embeddings for each text in the data and upsert the vectors to Pinecone.
    """
    tasks = []
    data = pd.read_csv(data_directory)
    batch_texts = []
    batch_metadatas = []
    for record in tqdm(data.itertuples()):
        texts = record.text.split()
        metadatas = [
            {
                "chunk": j,
                "text": text,
                "wiki-id": str(record.id),
                "source": record.url,
                "title": record.title,
            }
            for j, text in enumerate(texts)
        ]
        batch_texts.extend(texts)
        batch_metadatas.extend(metadatas)
        if len(batch_texts) >= BATCH_LIMIT:
            tasks.append(asyncio.ensure_future(process_batch((batch_texts, batch_metadatas))))
            batch_texts = []
            batch_metadatas = []
    if len(batch_texts) > 0:
        tasks.append(asyncio.ensure_future(process_batch((batch_texts, batch_metadatas))))
    await asyncio.gather(*tasks)

async def main():
    # Load data
    data_directory = 'data.csv'  # Replace with your actual data directory
    # Run the main function
    await send_embeddings_to_pinecone_async(data_directory)
    # Close the aiohttp session
    await session.close()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
