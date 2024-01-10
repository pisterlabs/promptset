from langchain.vectorstores.pgvector import PGVector

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# The connection to the database
CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver="psycopg2",
    host="localhost",
    port="5432",
    database="postgres",
    user="username",
    password="password",
)


def make_query(query, top_k):
    # The embedding function that will be used to store into the database
    embedding_function = SentenceTransformerEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Creates the database connection to our existing DB
    db = PGVector(
        connection_string=CONNECTION_STRING,
        collection_name="embeddings",
        embedding_function=embedding_function,
    )

    docs_with_scores = db.similarity_search_with_score(query, k=top_k)

    # print results
    for doc, score in docs_with_scores:
        print("-" * 80)
        print("Score: ", score)
        print(doc.page_content)
        print("-" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="query")
    parser.add_argument(
        "--top_k", type=int, default=2, help="how many similar entries to return"
    )

    args = parser.parse_args()

    make_query(query=args.query, top_k=args.top_k)
