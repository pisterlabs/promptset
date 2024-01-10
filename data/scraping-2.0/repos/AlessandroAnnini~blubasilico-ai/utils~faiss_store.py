import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import time


def create_store(docs, openai_api_key):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=openai_api_key
    )

    # create db
    print("Creating FAISS vector db...")
    db = FAISS.from_documents(docs, embeddings)
    print("Done.")

    return db


def create_multiple_and_merge(docs_batches, openai_api_key):
    db = None
    count = 0
    # for every batch create a db and if no db exists, create it, otherwise merge it
    for batch in docs_batches:
        # sleep for 65 seconds to avoid openai api limit
        time.sleep(65)
        count += 1
        print(f"Creating db for batch {count} of {len(docs_batches)}...")
        batch_db = create_store(batch["docs"], openai_api_key)
        if db is None:
            db = batch_db
        else:
            db.merge_from(batch_db)

    return db


def save_store(repo_name, db, overwrite=False):
    db_path = f"faiss_index/{repo_name}"

    if os.path.exists(db_path):
        if overwrite:
            # delete db directory
            print(f"Found existing vector db at {db_path}, deleting...")
            os.system(f"rm -rf {db_path}")
        else:
            raise ValueError(f"Vector db already exists at {db_path}.")

    # save db
    print(f"Saving vector db at {db_path}...")
    db.save_local(db_path)


def get_store(repo_names, openai_api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    db_array = []

    # load all dbs
    for repo_name in repo_names:
        print(f"Loading vector db for {repo_name}...")
        db_path = f"faiss_index/{repo_name}"

        if os.path.exists(db_path):
            print(f"Found existing vector db for {repo_name}, loading...")
            db = FAISS.load_local(db_path, embeddings)
            db_array.append(db)

    if len(db_array) == 0:
        raise ValueError("No db found for the given repo names.")

    # merge all dbs to the first one
    db = db_array[0]
    for i in range(1, len(db_array)):
        db.merge_from(db_array[i])

    return db
