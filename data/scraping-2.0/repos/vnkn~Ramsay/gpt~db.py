import ast
import hashlib
import json
import os
import re

import openai
import pandas as pd
import pinecone
from dotenv import load_dotenv
from gpt_format import format_web_clean, format_web_dirty, format_yt
from scrape_web import extract_from_web
from scrape_yt import extract_video_id, get_transcript
from sentence_transformers import SentenceTransformer

# MODEL = SentenceTransformer('all-MiniLM-L6-v2')
MODEL = SentenceTransformer('Dizex/FoodBaseBERT-NER')
# MODEL = SentenceTransformer('davanstrien/deberta-v3-base_fine_tuned_food_ner')

COLS = [
    "id",
    "url",
    "vegan",
    "appliances",
    "appliance_costs",
    "ingredients",
    "ingredient_costs",
    "steps",
    "serves",
    "time",
    "title",
]

FNAME = "data.csv"


def remove_flags_from_url(url):
    # Remove http:// or https:// from the beginning of the URL
    url = re.sub(r"^https?://", "", url)
    # Remove any flags or query parameters
    url = re.sub(r"\?.*$", "", url)
    # Remove trailing slash
    url = re.sub(r"\/$", "", url)
    # Normalize the URL
    url = re.sub(r"\/{2,}", "/", url)
    return url


def hash_url(url):
    url = remove_flags_from_url(url)
    md5_hash = hashlib.md5(url.encode()).hexdigest()
    return str(md5_hash)[:7]


def write_data(df):
    df = df[COLS]
    df.to_csv(FNAME, mode="a", header=False, index=False)


def parse_yt(url):
    transcript = get_transcript(url)
    data = format_yt(transcript)
    return data


def parse_web(url):
    text, clean = extract_from_web(url)
    data = format_web_clean(text) if clean else format_web_dirty(text)
    return data


def parse_url(url):
    hash = extract_video_id(url)
    is_yt = hash is not None
    hash = hash_url(url) if not hash else hash

    if hash in pd.read_csv(FNAME).id.values:
        data = read_data(hash)
    else:
        data = parse_yt(url) if is_yt else parse_web(url)
        data = pd.DataFrame.from_dict(data, orient="index").T

        data["id"] = hash
        data["url"] = url

        write_data(data)
    return data


def read_data(id):
    df = pd.read_csv(FNAME)
    df = df[df.id == id]
    # intreprets lists
    list_cols = (
        "appliances",
        "appliance_costs",
        "ingredients",
        "ingredient_costs",
        "steps",
    )
    for col in list_cols:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x))

    data = df.to_dict(orient="records")[0]
    return data


def init_db():
    if not os.path.isfile(FNAME):
        with open(FNAME, "w") as f:
            f.write(",".join(COLS) + "\n")


def pinecone_init() -> pinecone.Index:
    load_dotenv()
    PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
    pinecone.init(api_key=PINECONE_API_KEY, environment="us-west1-gcp-free")

    index_name = "ramsay-foodbasebert"
    # index_name = "ramsay-ada-002"
    # only create index if it doesn't exist
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            dimension=MODEL.get_sentence_embedding_dimension(),
            # dimension=1536,  # for text-embedding-ada-002
            metric="cosine",
        )

    # now connect to the index
    index = pinecone.GRPCIndex(index_name)
    return index


def create_embedding(text):
    return MODEL.encode([text])[0]

    # res = openai.Embedding.create(
    #     input=[text],
    #     engine="text-embedding-ada-002",
    # )
    # return res["data"][0]["embedding"]


def pinecone_insert(index: pinecone.Index, data: dict, metadata={}):
    id = data["id"]

    # ingredients = data["ingredients"]
    # ingredients = ",".join(ingredients)
    # data_to_embed = ingredients
    data_to_embed = data['title']

    embed = create_embedding(data_to_embed)
    index.upsert(vectors=[(id, embed, metadata)])


def pinecone_load_db(index: pinecone.Index):
    for _, row in pd.read_csv(FNAME).iterrows():
        pinecone_insert(index, row)


def pinecone_search(index, query, top_k=10):
    query_embedding = create_embedding(query)
    res = index.query(queries=[query_embedding], top_k=top_k)
    return res['results'][0]['matches']


if __name__ == "__main__":
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    init_db()
    index: pinecone.Index = pinecone_init()

    pinecone_load_db(index)

    res = pinecone_search(index, "pizza")

    for r in res:
        id = r['id']
        data = read_data(id)
        print(r['score'], data['title'])


    # pinecone_insert(index, read_data("ulhRORJpuBM"))

    # print(parse_url("https://youtu.be/ulhRORJpuBM"))

    # print(hash_url("https://cooking.nytimes.com/recipes/1021961-charleston-red-rice"))
