#Import libraries
import os
from struct import pack
import requests
import singlestoredb as s2


# List the variables
limit = int(os.getenv('LIMIT', '10')) # Set a limit on how many rows you want to read and write back
source_table = os.environ.get('SOURCE_TABLE', 'reviews_yelp') # Set which source table you want to read data from
source_table_PK= os.environ.get('SOURCE_TABLE_PK', 'review_id') # Set which column in the source table is the primary key
source_table_text_column = os.environ.get('SOURCE_TABLE_TEXT_COLUMN', 'text') # Set which column in the source table contains the text that you want to embed
destination_table = os.environ.get('DESTINATION_TABLE', 'reviews_yelp_embedding') # Set which destination table you will write data into - if that table doesn;t exist, we create it in the script
db_endpoint = os.environ.get('ENDPOINT', '') # Set the host string to SingleStoreDB. It should look like svc-XXXX.svc.singlestore.com
connection_port = os.environ.get('CONNECTION_PORT', '3306') # Set the port to access that endpoint. By default it is 3306
username = os.environ.get('USERNAME', '') # Set the username to access that endpoint
password = os.environ.get('PASSWORD', '') # Set the password for the username to access that endpoint
database_name = os.environ.get('DATABASE_NAME', '') # Set the database name you want to access
API_KEY = os.environ.get('OPENAPI_API_KEY', '') # Set the API Key from OpenAI
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', '') # Define which OpenAI model to use
URL_EMBEDDING = os.environ.get('URL', 'https://api.openai.com/v1/embeddings') # URL to access OpenAI
BATCH_SIZE = os.environ.get('BATCH_SIZE', '2000') # Set how many rows you want to process per batch

#Configure Header and connections
HEADERS = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json',
}

fetch_conn = s2.connect(host=db_endpoint, user=username, password = password, database=database_name)
insert_conn = s2.connect(host=db_endpoint, user=username, password = password, database=database_name)

# Lambda function
def handler(event, context):
    fetch_cur = fetch_conn.cursor()
    
    # Create Table if not existent
    query_create_table = '''
    CREATE TABLE IF NOT EXISTS {} (
    {} text, embedding blob, batch_index int, usage_tokens_batch int, timestamp datetime, model text)'''.format(destination_table,source_table_PK)
    fetch_cur.execute(query_create_table)
    
    # Select rows with text that has no embeddings
    query_read = 'select {}, {} from {} where {} NOT IN (select {} from {}) limit %s'.format(source_table_PK,source_table_text_column, source_table,source_table_PK,source_table_PK, destination_table)
    fetch_cur.execute(query_read, (limit,))
    
    fmt = None

    while True:
        
         #Create the embeddings by calling URL
        rows = fetch_cur.fetchmany(BATCH_SIZE)
        if not rows: break

        res = requests.post(URL_EMBEDDING,
                            headers=HEADERS,
                            json={'input': [row[1].replace('\n', ' ')
                                            for row in rows],
                                  'model': EMBEDDING_MODEL}).json()

        if fmt is None:
            fmt = '<{}f'.format(len(res['data'][0]['embedding']))
        
        #Insert the embeddings into destination table
        
        insert_embedding = 'INSERT INTO {} ({}, embedding,batch_index,usage_tokens_batch,timestamp,model) VALUES (%s, %s, %s,%s,now(),%s)'.format(destination_table,source_table_PK)
        data = [(row[0], pack(fmt, *ai['embedding']), ai['index'], res['usage']['total_tokens'], EMBEDDING_MODEL) for row, ai in zip(rows, res['data'])]

        insert_conn.cursor().executemany(insert_embedding, data)
