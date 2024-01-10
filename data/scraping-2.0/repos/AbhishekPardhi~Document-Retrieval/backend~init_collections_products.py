import sys
sys.path.append("..")
import os.path
from tqdm import tqdm
import pandas
import time
import qdrant_client
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings

from backend.config import DATA_DIR, QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME

def upload_embeddings():
    # Create client
    client = qdrant_client.QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=True,
    )

    # create collection
    vectors_config = qdrant_client.http.models.VectorParams(
        size=1536,
        distance=qdrant_client.http.models.Distance.COSINE,
    )

    # Load data
    file_path = os.path.join(DATA_DIR, 'bigBasketProducts.csv')

    df = pandas.read_csv(file_path)
    metadatas = [{'source':int(df.loc[i][0]), 'row':i} for i in range(len(df))]
    df = df.apply(lambda x: x.to_json(), axis=1)

    docs = [row for row in df]

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=vectors_config,
    )

    # create vector store
    embeddings = OpenAIEmbeddings()

    vector_store = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
    )

    retries_dict = {}

    for i in range(0, len(docs), 32):
        try:
            vector_store.add_texts(
                texts=docs[i:i+32],
                metadatas=metadatas[i:i+32],
                ids=tqdm.tqdm(range(i, i+32))
            )
        except Exception as e:
            print(i, e)
            i = i - 32
            retries_dict[i] = retries_dict.get(i, 0) + 1
            if retries_dict[i] > 5:
                print(f"Failed to add documents at index {i} after 3 retries. Skipping...")
                i += 32
                continue
            time.sleep(1)


if __name__ == '__main__':
    upload_embeddings()