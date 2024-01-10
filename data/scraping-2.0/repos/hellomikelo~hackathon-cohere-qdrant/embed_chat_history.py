import discord
import pandas as pd
import os
import time_uuid

import cohere
from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.http import models as rest

from dotenv import load_dotenv

QDRANT_CLOUD_HOST = "19531f2c-0717-4706-ac90-bd8dd1a6b0cc.us-east-1-0.aws.cloud.qdrant.io"
QDRANT_COLLECTION_NAME = 'discord'
# Google Drive path
CHAT_HISTORY_PATH = '/content/drive/MyDrive/career/projects/hackathons/lablab-cohere-qdrant-hackathon/discord-chat-history.csv'

load_dotenv()

cohere_client = cohere.Client(os.getenv('COHERE_API_KEY'))
qdrant_client = QdrantClient(
    host=QDRANT_CLOUD_HOST, 
    prefer_grpc=False,
    api_key=os.getenv('QDRANT_API_KEY'),
)

    
def clean_chat(df):
    """Clean chat history to keep only alphanums and Han Ideographs."""
    _df = df.copy()
    _df['content'] = (_df['content']
                      .str.replace('[^a-zA-Z\u4E00-\u9FFF\s]', '', regex=True)
                      .str.replace('(http\w+|\n)', '', regex=True)
                      .str.replace('<.*>', '', regex=True)
                      .str.lower()
                      .str.strip()
                      .fillna('')
                      )
    _df['id'] = _df.created_at.apply(lambda x: str(time_uuid.TimeUUID.with_utc(pd.to_datetime(x))))
    _df['word_count'] = _df.content.apply(lambda x: len(x.split(' ')))
    return _df


def create_embeddings(dataset: pd.DataFrame):

    # Embed chat messages
    embeddings = cohere_client.embed(
        texts=dataset.content.tolist(),
        model='multilingual-22-12',
    )

    vector_size = len(embeddings.embeddings[0])
    vectors = [list(map(float, vector)) for vector in embeddings.embeddings]

    ids = dataset.id.tolist()

    # Create Qdrant vector database collection
    qdrant_client.recreate_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=vector_size, 
            distance=rest.Distance.DOT # for multilingual model
            # distance=rest.Distance.COSINE # for large model
        ),
    )

    # Upsert new embeddings into vector search engine
    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME, 
        points=rest.Batch(
            ids=ids,
            vectors=vectors,
            payloads=dataset.to_dict(orient='records'),
        )
    )

    print('Vector database created.')


def test_embed():
    # Test query embeddings
    new_embeddings = cohere_client.embed(
        texts=["discussions on horses", "discussions on asian countries", "interesting dog facts"],
        # model="large",
        model='multilingual-22-12',
    )

    results = []
    k_max = 5

    new_vectors = [list(map(float, vector)) for vector in new_embeddings.embeddings]

    for embedding in new_vectors:
        response = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=embedding,
            limit=k_max,
        )
        results.append([record.payload['content'] for record in response])
    print(results)


if __name__ == '__main__':
    df = pd.read_csv(CHAT_HISTORY_PATH, index_col=0)
    dataset = clean_chat(df)
    embed = True
    if embed:
        create_embeddings(dataset)
