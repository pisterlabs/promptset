import openai
import pinecone
import sys
import os
import pandas
from dotenv import load_dotenv

load_dotenv()
pinecone.init(api_key=os.getenv("PINECONE_API_KEY") or "", environment=os.getenv("PINECONE_ENV") or "")
openai.api_key = f"{os.getenv('OPENAI_API_KEY')}" + f"{os.getenv('OPENAI_API_KEY_2')}"

pinecone_index = pinecone.Index(os.getenv("PINECONE_INDEX") or "")

def read_csv(path):
    df = pandas.read_csv(path, encoding="utf-8", sep=";", names=["text", "link"])
    return df

def embedding(df):
    texts = []
    metadata = []
    for col, row in df.iterrows():
        texts.append(row.text)
        metadata.append({"text": row.text, "link": row.link})
    res = openai.Embedding.create(input=texts, engine="text-embedding-ada-002")
    embeddings = [vec["embedding"] for vec in res["data"]]
    return embeddings, metadata

def upsert(df):
    embeds, meta = embedding(df)
    try:
        v = [{"id": str(n), "values": vec, "metadata": meta[n]} for n, vec in enumerate(embeds)]
        pinecone_index.upsert(vectors=v)
        print("Upsert successful")
    except Exception as e:
        print("Upsert failed")
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    df = read_csv(os.path.join(os.path.dirname(__file__), "..", "..", "DataSet.csv"))
    upsert(df)