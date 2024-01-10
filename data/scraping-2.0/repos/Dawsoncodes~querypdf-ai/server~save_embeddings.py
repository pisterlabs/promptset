import argparse
import shutil
import os
from tqdm import tqdm
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
from data.get_collections import get_collections
from data.format_docs import format_document
from lib.chroma import chroma_client, client, filter_ids_if_exists

load_dotenv()


def save_embeddings(reset_db: bool):
    """
    - Save all embeddings to the database, optionally resetting the database
    - If --reset is passed when running the script from the command line, the database will be reset.
    - This function is not used in the server, but is used to generate the embeddings for the server on startup
    and it might take some time to finish.

    :param reset_db: Whether or not to reset the database
    """

    base_dir = os.path.join(os.path.dirname(__file__), "data", "formatted")

    if reset_db:
        print("Resetting database...")
        client.reset()

    if os.path.exists(base_dir):
        print("Removing old data...")
        shutil.rmtree(base_dir)

    print("Formatting documents...")
    format_document()

    collections = get_collections()

    print(f"Found {len(collections)} collections")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

    chroma_collection_names = [col.name for col in client.list_collections()]

    for collection in collections:
        items_in_dir = os.listdir(f"{base_dir}/{collection}")
        items_count = len(items_in_dir)

        print(f"Found {items_count} items in {collection}")

        for i, item in enumerate(tqdm(items_in_dir, desc=f"Processing {collection}")):
            print(f"({i+1}/{items_count})")

            loader = TextLoader(f"{base_dir}/{collection}/{item}", "utf-8")
            documents = loader.load()

            docs = text_splitter.split_documents(documents)
            ids = [f"{item.split('.')[0]}-{i}" for i, _ in enumerate(docs)]

            filtered_ids = filter_ids_if_exists(
                collection, ids, chroma_collection_names)

            if len(filtered_ids) < 1 or len(filtered_ids) != len(ids):
                print(f"Skipping ids for {collection}:",
                      ", ".join([id for id in ids]))
                continue

            chroma_client.from_documents(docs, collection_name=collection,
                                         ids=filtered_ids, embedding=OpenAIEmbeddings(), client=client)

    print("Generated all embeddings for collections:\n-",
          "\n- ".join(collections))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Reset the database")
    args = parser.parse_args()

    save_embeddings(args.reset)
