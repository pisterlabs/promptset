"""adds documents in supabase vector database"""
from threading import Thread
import json
from typing import List
from os import listdir, environ
import openai
from supabase import create_client
from pprint import pprint
# thread
MAX_NUM_OF_THREADS = 8

# open ai details
openai.api_key = environ["OPENAI_API_KEY"]
# supabase details
supabase_url = environ["SUPABASE_URL"]
supabase_key = environ["SUPABASE_KEY"]
supabase = create_client(supabase_url=supabase_url, supabase_key=supabase_key)

COLLECTION_JSON = "compiled.json"


def compile_all_documents(path: str) -> None:
    """gets all vector documents and generates a compiled file for supabase loading"""
    documents = {"documents": []}
    for this_file in listdir(f"./{path}"):
        with open(f"./{path}/{this_file}", "r", encoding="utf-8") as file:
            obj = json.load(file)
            docs = obj["documents"]
            srcs = obj["sources"]
            if len(docs) == len(srcs):
                documents["documents"] += [{"content": docs[i], "source":srcs[i]}
                                           for i in range(len(docs))]
    with open(COLLECTION_JSON, "w", encoding="utf-8") as file:
        json.dump(documents, file)


def write_embeddings_to_documents(documents: List[object]) -> None:
    """writes all documents and embeddings to supabase documents table"""

    for document in documents:
        embedding = openai.Embedding.create(
            input=document["content"], model="text-embedding-ada-002"
        )["data"][0]["embedding"]
        document = {"content": document["content"],
                    "embedding": embedding, "source": document["source"]}

        supabase.table("documents").insert(document).execute()


# def segment_write_to_supabase(all_docs:List[object])->None:

def add_new_user(chat_id: str, username: str) -> None:
    """Adds new user into db"""
    try:
        supabase.table("users").insert({
            "id": chat_id,
            "username": username
        }).execute()
    except:
        pass


def remove_user(chat_id: str) -> None:
    """Remove users from db"""
    try:
        supabase.table("users").delete().eq("id", chat_id).execute()
    except:
        pass


def segment_content(documents: List[object], num_of_threads: int) -> List[List[object]]:
    """breaks documents into chunks"""
    segments = [[] for i in range(num_of_threads)]
    for i, document in enumerate(documents):
        multiplier = i // num_of_threads
        segments[i - num_of_threads*multiplier].append(document)
    return segments


def segment_write_to_supabase(documents: List[object]) -> None:
    """threaded write to supabase"""
    threads = [Thread(target=write_embeddings_to_documents, kwargs={"documents": segment})
               for segment in segment_content(documents, MAX_NUM_OF_THREADS)]
    # START THREADS
    for thread in threads:
        thread.start()
    # JOIN THREADS
    for thread in threads:
        thread.join()


def get_context_from_supabase(query: str, threshold: float, count: int) -> str:
    """get contexts from supabase"""
    contexts = []
    embedding = openai.Embedding.create(
        input=query, model="text-embedding-ada-002")["data"][0]["embedding"]
    response = supabase.rpc("match_documents", {
        "query_embedding": embedding,
        "similarity_threshold": threshold,
        "match_count": count,
    }).execute()
    for context in response.data:
        content = context["content"]
        source = context["source"]
        line = f"{content} (source: {source})"
        contexts.append(line)
    return "\n".join(contexts)

# def add_message_to_supabase(chat_id:str, message_id:str, message:str)->None:


if __name__ == "__main__":
    # compile_all_documents("vector_documents")
    # with open(COLLECTION_JSON, "r", encoding="utf-8") as file:
    #     obj = json.load(file)
    #     docs = obj["documents"]
    # segment_write_to_supabase(docs)
    pass
