import openai
import numpy as np
import sys
from scipy.spatial.distance import cosine
import psycopg2


openai.api_key = ''

def etreive ():
    try:
        connection = psycopg2.connect(user="postgres",
                                    password="postgres",
                                    host="localhost",
                                    port="5432",
                                    database="regtech")
        cursor = connection.cursor()
        postgreSQL_select_Query = "select embedding, name from specialist"

        cursor.execute(postgreSQL_select_Query)
        mobile_records = cursor.fetchall()
        emds = []
        for row in mobile_records:
            emds.append( [row[0], row[1]])
        return emds
    except (Exception, psycopg2.Error) as error:
        err = error

    finally:
        # closing database connection.
        if connection:
            cursor.close()
            connection.close()

def encode_text(text):
    response = openai.Embedding.create(
            input = text,
            model = 'text-embedding-ada-002',
            max_tokens=2046,
    )
    embeddings = response['data'][0]['embedding']
    return embeddings

query_text = sys.argv[1]
query_embedding = encode_text(query_text)
collection_embeddings = [embedding[0] for embedding in etreive()]
similarity_scores = [1 - cosine(list(map(float,query_embedding)), list(map(float,emb))) for emb in collection_embeddings]


collection_texts = [name[1] for name in etreive()]
sorted_texts = [text for text in sorted(zip(similarity_scores, collection_texts), reverse=True)]
sorted_scores = sorted(similarity_scores, reverse=True)

result = []
for text, score in zip(sorted_texts, sorted_scores):
    result.append(text)


sys.stdout.write(str(result))
