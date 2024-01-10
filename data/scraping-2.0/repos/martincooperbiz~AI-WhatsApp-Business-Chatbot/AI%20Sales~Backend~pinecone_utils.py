import pinecone
import pandas as pd
import openai
import os

from dotenv import load_dotenv
load_dotenv()

# pinecone.init(api_key=os.getenv("PINECONE_API_KEY"))
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
# pinecone.delete_index("hormozigpt")
print("delete complete")
# print(os.getenv("PINECONE_API_KEY"), os.getenv("PINECONE_ENV"))

print(pinecone.list_indexes())


# pinecone.create_index(
#     os.getenv("PINECONE_INDEX_NAME"), 
#     dimension=1536, 
#     metric="cosine", 
#     pod_type="p1"
# )

index = pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))
# print(index.describe_index_stats())
print("init complete")

df = pd.read_csv("Backend/history.csv")
print(df.head().count())
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )

    return response['data'][0]['embedding']


def addData(index, context):
    my_id = index.describe_index_stats()['total_vector_count']

    chunkInfo = (str(my_id),
                 get_embedding(context),
                 {'context':context})

    index.upsert(vectors=[chunkInfo])


for indexx, row in df.iterrows():
    # addData(index, row["content"])
    print("add data: ", row["content"])


print("Completed")