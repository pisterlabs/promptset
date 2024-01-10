import os
from langchain.vectorstores import Chroma
from llm.embedding_manager import set_embedding

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))

INDEX_NAME_1 = "index_1"
INDEX_PATH_1 = f"{parent_dir}/db_chroma/db_1"
INDEX_PATH_2 = "vector_db/faiss_1"


def start_chroma_db():
    vector_db = Chroma(
        collection_name=INDEX_NAME_1,
        embedding_function=set_embedding(),
        persist_directory=INDEX_PATH_1,
    )
    return vector_db


# import os
# from langchain.vectorstores import FAISS

# texts_1 = ["La filosofía es rascarse cuando no pica."]
# texts_2 = ["El arte no tiene por qué entenderse a la primera."]
# faiss_text = FAISS.from_texts(texts=texts_1, embedding=set_embedding())

# def start_faiss_db():
#     if not os.path.exists(INDEX_PATH_2):
#         os.makedirs(INDEX_PATH_2)
#         faiss_text.save_local(
#             folder_path=INDEX_PATH_2,
#         )
#         print(f"Embeddings saved in: {INDEX_PATH_2}")
#         return True
#     else:
#         local_db = FAISS.load_local(
#             folder_path=INDEX_PATH_2, embeddings=set_embedding()
#         )
#         local_db.merge_from(faiss_text)
#         local_db.save_local(folder_path=INDEX_PATH_2)
#         print(f"Loaded db: {local_db}")
#         return True
