from psycopg2 import connect
from pgvector.psycopg2 import register_vector
import openai

def set_openai_key(config):
    openai.api_key = config.get("OpenAI", "key")
    
def setup_vector(connection):
    register_vector(connection) 

def postgreSQL_connect(config):
    host = config.get("PostgreSQL", "host")
    database = config.get("PostgreSQL", "database")
    user = config.get("PostgreSQL", "user")
    password = config.get("PostgreSQL", "password")

    connection = connect(
        host=host, 
        database=database, 
        user=user, 
        password=password
    )
    setup_vector(connection)
    cursor = connection.cursor()

    return connection, cursor

def postgreSQL_disconnect(connection, cursor):
    cursor.close()
    connection.close()
    