import chromadb
from chromadb.config import Settings
from constant.frontend_operation_enum import FrontendOperation
from constant.frontend_operation_param_enum import FrontendOperationParam
from langchain.embeddings.openai import OpenAIEmbeddings
from config.vectordb_config import FRONTEND_OPERATION_COLLECTION


def update_frontend_operation_embeddings(client, collection):
    embeddings = OpenAIEmbeddings().embed_documents(
        FrontendOperation.all_descriptions()
    )
    collection.upsert(
        embeddings=embeddings,
        ids=FrontendOperation.all_function_names(),
        documents=FrontendOperation.all_descriptions(),
        metadatas=FrontendOperation.all_metadata(),
    )
    print(collection.count())


def update_frontend_operation_param_embeddings(client, collection):
    embeddings = OpenAIEmbeddings().embed_documents(
        FrontendOperationParam.all_descriptions()
    )
    collection.upsert(
        embeddings=embeddings,
        ids=FrontendOperationParam.all_function_names(),
        documents=FrontendOperationParam.all_descriptions(),
        metadatas=FrontendOperationParam.all_metadata(),
    )
    print(collection.count())


client = chromadb.Client(
    Settings(chroma_db_impl="duckdb+parquet", persist_directory="./vectordb")
)


collection = client.create_collection(FRONTEND_OPERATION_COLLECTION)
update_frontend_operation_embeddings(client, collection)
update_frontend_operation_param_embeddings(client, collection)

# collection = client.get_collection(FRONTEND_OPERATION_COLLECTION)
# query_embeddings = OpenAIEmbeddings().embed_documents(["巡检"])
# results = collection.query(
#     n_results=1,
#     query_embeddings=query_embeddings,
#     where={"last_operation": FrontendOperation.ADD_INSPECTION_STRATEGY.function_name},
# )
# print(results)
