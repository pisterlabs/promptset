import tiktoken
import os
import openai
import dotenv
import numpy as np
import time

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PROJECT = os.getenv("PROJECT_NAME")
assert OPENAI_API_KEY and PROJECT
openai.api_key = OPENAI_API_KEY

enc = tiktoken.get_encoding("cl100k_base")


def get_size(text):
    return len(enc.encode(text))


## vector search from make_vecs_from_json/main.py
# It would be better to make library modules for this.
def embed_texts(texts, sleep_after_sucess=1):
    EMBED_MAX_SIZE = 8150  # actual limit is 8191
    if isinstance(texts, str):
        texts = [texts]
    assert all(text != "" for text in texts)
    for i, text in enumerate(texts):
        text = text.replace("\n", " ")
        tokens = enc.encode(text)
        if len(tokens) > EMBED_MAX_SIZE:
            text = enc.decode(tokens[:EMBED_MAX_SIZE])
        texts[i] = text

    while True:
        try:
            res = openai.Embedding.create(input=texts, model="text-embedding-ada-002")
            time.sleep(sleep_after_sucess)
        except Exception as e:
            print(e)
            time.sleep(1)
            continue
        break

    return res


def embed(text, sleep_after_sucess=0):
    # short hand for single text
    r = embed_texts(text, sleep_after_sucess=sleep_after_sucess)
    return r["data"][0]["embedding"]


def get_sorted(vindex, query):
    q = np.array(embed(query, sleep_after_sucess=0))
    buf = []
    for body, (v, payload) in vindex.items():
        buf.append((q.dot(v), body, payload))
    buf.sort(reverse=True)
    return buf


#
## vector search from make_vecs_from_json/main.py
