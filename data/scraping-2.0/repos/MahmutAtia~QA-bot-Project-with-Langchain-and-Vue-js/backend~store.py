from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings import HuggingFaceInstructEmbeddings

CONNECTION_STRING = "postgresql+psycopg2://postgres:test@localhost:5432/vector_db"
COLLECTION_NAME = "state_of_the_union_test"


def get_store():

    # Embedding
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                        # model_kwargs={"device": "cuda"} 
                                                        )

    # Vector Store
    store = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)
 
    return  store
