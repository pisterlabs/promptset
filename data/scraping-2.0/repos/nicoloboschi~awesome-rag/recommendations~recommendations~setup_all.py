import csv

from astrapy.db import AstraDB
from astrapy.utils import http_methods
from langchain.schema.vectorstore import VectorStore

from recommendations.recommendations.util import create_astra_vector_store, create_raw_astra_client, \
    PRODUCTS_COLLECTION_NAME, USERS_COLLECTION_NAME, RECOMMENDATIONS_COLLECTION_NAME


def index(vector: VectorStore):
    texts = []
    metadatas = []
    ids = []
    with open("./products.csv") as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter=';', quotechar='"')
        for i, row in enumerate(csv_reader):
            texts.append(row["DESCRIPTION"])
            metadatas.append({"NAME": row["NAME"], "PRICE": row["PRICE"], "DESCRIPTION": row["DESCRIPTION"], "ID": i})
            ids.append(i)
    vector.add_texts(texts, metadatas, ids=ids)


def create_table(astra_db: AstraDB, table: str):

    response = astra_db._request(
        method=http_methods.POST,
        path=astra_db.base_path,
        json_data={"createCollection": {"name": table}},
    )
    print(response)


if __name__ == '__main__':
    # astra_vector_store = create_astra_vector_store(PRODUCTS_COLLECTION_NAME)
    # astra_vector_store.delete_collection()
    astra_vector_store = create_astra_vector_store(PRODUCTS_COLLECTION_NAME)
    #
    index(astra_vector_store)
    print("indexed products")

    astra_db = create_raw_astra_client()
    create_table(astra_db, RECOMMENDATIONS_COLLECTION_NAME)
    print(f"{RECOMMENDATIONS_COLLECTION_NAME} collection created")
    #
    # astra_db.collection("users").insert_one({
    #     "handle": "max",
    #     "first_name": "Max",
    #     "last_name": "Doe",
    #     "age": 30,
    #     "gender": "male"
    # })
    #
    # astra_db.collection("users").insert_one({
    #     "handle": "adele",
    #     "first_name": "Adele",
    #     "last_name": "Doe",
    #     "age": 54,
    #     "gender": "female"
    # })
    # print("inserted users")
