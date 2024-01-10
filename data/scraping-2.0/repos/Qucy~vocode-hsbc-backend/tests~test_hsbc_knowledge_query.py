import os
import psycopg2
import openai

from dotenv import load_dotenv

# load environment variables
load_dotenv()

# Create database connection
host = os.getenv('PG_HOST')
dbname = os.getenv('PG_DB_NAME')
user = os.getenv('PG_USER')
password = os.getenv('PG_PASSWORD')
sslmode = os.getenv('PG_SSLMODE')

# openai configuration
openai.api_key = os.getenv('AZURE_OPENAI_API_KEY')
openai.api_version = os.getenv('AZURE_OPENAI_API_VERSION')
openai.api_type = os.getenv('AZURE_OPENAI_API_TYPE')
openai.api_base = os.getenv('AZURE_OPENAI_API_BASE')


def retrieve_hsbc_knowledge_pgvector(input: str):
    """
    Retrieve HSBC knowledge from pgvector
    """
    # Construct connection string
    conn_string = f"host={host} user={user} dbname={dbname} password={password} sslmode={sslmode}"
    conn = psycopg2.connect(conn_string)

    assert conn is not None, "Database connection failed."

    # get embedding from input
    response = openai.Embedding.create(input=input, engine="text-embedding-ada-002")
    embeddings = response['data'][0]['embedding']

    assert embeddings is not None, "Embedding failed."

    # create cursor
    cur = conn.cursor()
    # execute query
    cur.execute(f"SELECT content FROM hsbc_homepage_content ORDER BY embedding <-> '{embeddings}' LIMIT 3;")
    # retrieve records
    records = cur.fetchall()
    # close cursor
    cur.close()
    # print top n records
    for record in records:
        print("========================================")
        print(record[0])


if __name__ == '__main__':
    # test retrieve_hsbc_knowledge_pgvector  TODO looks like the accurcy is not good enough, need to be further improved
    retrieve_hsbc_knowledge_pgvector("opening mobile account in 5 mintues")

    