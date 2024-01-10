import openai
import sys
import json
import os
import psycopg2
import time


PG_URL = os.environ.get("PG_URL")

def compute_embeddings(texts):
    response =openai.Embedding.create(
        model="text-embedding-ada-002",
        input=texts
    )
    return [d["embedding"] for d in response.data]

def main():
    vector = compute_embeddings([sys.argv[1]])[0]
    # print(json.dumps(vector))
    with psycopg2.connect(PG_URL) as conn:
        with conn.cursor() as cur:
            start = time.time()
            print(f"EXPLAIN ANALYZE SELECT sentence FROM sentences WHERE isPublic = true ORDER BY embedding <=> '{json.dumps(vector)}' LIMIT 5")
            #cur.execute("SELECT sentence FROM sentences ORDER BY embedding <=> %s LIMIT 5", (json.dumps(vector),))
            #results = cur.fetchall()
            #print(f"Search took {time.time() - start} seconds")
            #for result in results:
            #    print(f'{result[0]}')


if __name__ == "__main__":
    main()