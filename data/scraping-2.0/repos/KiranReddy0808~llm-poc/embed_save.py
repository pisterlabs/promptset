from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
import os
from ast import literal_eval
import time

embeddings = HuggingFaceEmbeddings()


if not os.path.exists('./output.csv') :
    dataset = pd.read_csv("bigBasketProducts.csv")
    dataset_df = pd.DataFrame(dataset)

    for v in ['product', 'category', 'sub_category', 'brand', 'sale_price', 'market_price', 'type', 'rating', 'description']:
        print("embeddding " + v + " started")
        start = time.time()
        dataset_df[v + '_embedding'] = dataset_df[v].apply(
            lambda x: embeddings.embed_query(str(x))
        )
        end = time.time()
        print(f'embeddding {v} ended. Moving to next one. Runtime: {end - start}s')


    dataset_df.to_csv('output.csv', index = False)
    print("File output.csv created with embeddings.")

else:
    print("Reading output csv file")
    dataset_df = pd.DataFrame(pd.read_csv("output.csv"))
    for v in ['product', 'category', 'sub_category', 'brand', 'sale_price', 'market_price', 'type', 'rating', 'description']:
        dataset_df[v + '_embedding'] = dataset_df[v + '_embedding'].apply(literal_eval)
    print("Completed reading output csv file")

client = QdrantClient(host='localhost', port=6333)


vector_size = len(dataset_df["description_embedding"][0])

client.recreate_collection(
    collection_name="Products",
    vectors_config={
        "product": VectorParams(
            distance= Distance.COSINE,
            size=vector_size,
        ),
        "category": VectorParams(
            distance= Distance.COSINE,
            size=vector_size,
        ),
        "sub_category": VectorParams(
            distance= Distance.COSINE,
            size=vector_size,
        ),
        "brand": VectorParams(
            distance= Distance.COSINE,
            size=vector_size,
        ),
        "sale_price": VectorParams(
            distance= Distance.COSINE,
            size=vector_size,
        ),
        "market_price": VectorParams(
            distance= Distance.COSINE,
            size=vector_size,
        ),
        "type": VectorParams(
            distance= Distance.COSINE,
            size=vector_size,
        ),
        "rating": VectorParams(
            distance= Distance.COSINE,
            size=vector_size,
        ),
        "description": VectorParams(
            distance= Distance.COSINE,
            size=vector_size,
        ),
    }
)

frame_len = len(dataset_df.index)
fract = int(frame_len/100)
rem = frame_len%100
ind = 0
while ind <= fract:
    if ind < fract:
        mini_dataset_df = dataset_df.iloc[ind*100:ind*100 + 99]
    if ind == fract:
        mini_dataset_df = dataset_df.iloc[ind*100: ind*100 + rem]

    client.upsert(
        collection_name="Products",
        points=[
            PointStruct(
                id= int(v["index"])-1,
                vector={
                    "product": v["product_embedding"],
                    "category": v["category_embedding"],
                    "sub_category": v["sub_category_embedding"],
                    "brand": v["brand_embedding"],
                    "sale_price": v["sale_price_embedding"],
                    "market_price": v["market_price_embedding"],
                    "type": v["type_embedding"],
                    "rating": v["rating_embedding"],
                    "description": v["description_embedding"],
                },
                payload=v[["index", "product"]].to_dict(),
            )
            for k, v in mini_dataset_df.iterrows()
        ],
    )
    ind += 1

print("completed saving vectors in qdrant db")

