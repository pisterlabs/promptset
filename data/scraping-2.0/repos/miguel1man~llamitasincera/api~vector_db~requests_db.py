from langchain.vectorstores import Chroma
from llm.embedding_manager import set_embedding
from vector_db.manager_db import INDEX_NAME_1, INDEX_PATH_1

RESULTS = 2


def vector_db_query(
    query_text, collection=INDEX_NAME_1, directory=INDEX_PATH_1, results=RESULTS
):
    vectordb = Chroma(
        collection_name=collection,
        embedding_function=set_embedding(),
        persist_directory=directory,
    )

    docs = vectordb.similarity_search_with_score(query=query_text, k=results)
    return docs


""" 
question = vector_db_query("La filosofía es...")
print("answer:", question)
 """
