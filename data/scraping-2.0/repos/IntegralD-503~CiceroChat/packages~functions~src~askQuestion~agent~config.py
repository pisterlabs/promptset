import os
from dotenv import load_dotenv
from langchain.vectorstores.pgvector import PGVector

load_dotenv()

class EmbeddingsStore():
    def __init__(self):
        """initialize  connection """
        self.PGVECTOR_PASSWORD = os.environ['PGVECTOR_PASSWORD']
        self.PGVECTOR_USER = os.environ['PGVECTOR_USER']
        self.PGVECTOR_HOST = os.environ['PGVECTOR_HOST']
        self.PGVECTOR_PORT = os.environ['PGVECTOR_PORT']

    def createConnectionString(self, databaseName: str):
        self.connectionString = PGVector.connection_string_from_db_params(
            driver="psycopg2",
            host=self.PGVECTOR_HOST,
            port=int(self.PGVECTOR_PORT),
            database=databaseName,
            user=self.PGVECTOR_USER,
            password=self.PGVECTOR_PASSWORD,
        )

        return self.connectionString