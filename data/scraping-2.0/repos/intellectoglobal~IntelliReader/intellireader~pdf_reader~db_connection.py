from langchain.vectorstores.pgvector import PGVector
import os      
       
# # Alternatively, you can create it from environment variables.
CONNECTION_STRING = PGVector.connection_string_from_db_params(
driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
host=os.environ.get("PGVECTOR_HOST", "localhost"),
port=int(os.environ.get("PGVECTOR_PORT", "5434")),
database=os.environ.get("PGVECTOR_DATABASE", "pgvector_test"),
user=os.environ.get("PGVECTOR_USER", "postgres"),
password=os.environ.get("PGVECTOR_PASSWORD", "12345"),
)
