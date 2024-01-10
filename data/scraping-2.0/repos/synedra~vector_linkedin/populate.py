from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import json
import os
from getpass import getpass
import openai
from uuid import uuid4

try:
    from google.colab import files
    IS_COLAB = True
except ModuleNotFoundError:
    IS_COLAB = False

ASTRA_DB_SECURE_BUNDLE_PATH = os.environ["ASTRA_DB_SECURE_BUNDLE_PATH"] 

ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
ASTRA_DB_KEYSPACE = "vector"

cluster = Cluster(
    cloud={
        "secure_connect_bundle": ASTRA_DB_SECURE_BUNDLE_PATH,
    },
    auth_provider=PlainTextAuthProvider(
        "token",
        ASTRA_DB_APPLICATION_TOKEN,
    ),
)
session = cluster.connect()
keyspace = ASTRA_DB_KEYSPACE 

delete_table_statement = f"""DROP TABLE IF EXISTS vector.shakespeare_cql;"""
session.execute(delete_table_statement)

create_table_statement = f"""CREATE TABLE IF NOT EXISTS vector.shakespeare_cql (
    dataline INT PRIMARY KEY,
    play TEXT,
    player TEXT,
    playerline TEXT,
    embedding_vector VECTOR<FLOAT, 1536>
);"""

session.execute(create_table_statement)
print("Created table.")

create_vector_index_statement = f"""CREATE CUSTOM INDEX IF NOT EXISTS idx_embedding_vector
    ON {keyspace}.shakespeare_cql (embedding_vector)
    USING 'org.apache.cassandra.index.sai.StorageAttachedIndex'
    WITH OPTIONS = {{'similarity_function' : 'dot_product'}};
"""

session.execute(create_vector_index_statement)
print("Created index.")

openai.api_key=os.environ["OPENAI_API_KEY"]

embedding_model_name = "text-embedding-ada-002"

quote_array = json.load(open("/Users/kirstenhunter/Downloads/shakespeare.json"))

prepared_insertion = session.prepare(
    f"INSERT INTO vector.shakespeare_cql (dataline, player, playerline, embedding_vector) VALUES (?,?,?,?)"
)

# 
for index in range(len(quote_array)):
    if quote_array[index]["Play"] != "Romeo and Juliet":
        continue
    quote_id = quote_array[index]["Dataline"]
    previous_quote = ""
    next_quote = ""
    if index > 0:
        previous_quote = quote_array[index-1]["PlayerLine"]
    if index < len(quote_array):
        next_quote = quote_array[index+1]["PlayerLine"]
    quote_input = previous_quote + "\n" + quote_array[index]["PlayerLine"] + "\n" + next_quote
    result = openai.Embedding.create(
        input=quote_input,
        engine=embedding_model_name
    )

    session.execute(
        prepared_insertion,
        (quote_id, quote_array[index]["Player"], quote_array[index]["PlayerLine"], result.data[0].embedding)
    )

    print(quote_array[index]["Player"] + " : " + quote_array[index]["PlayerLine"])

