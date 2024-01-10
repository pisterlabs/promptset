import os
import requests
from pymilvus import connections, DataType, Collection, CollectionSchema, FieldSchema
from pymilvus_orm import connections, FieldSchema, CollectionSchema, Collection
from openai import embeddings
import json
import logging


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", 19530)

headers_openai = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}",
}


def check_argument(arg, expected_type):
    if not isinstance(arg, expected_type):
        raise TypeError(
            f"Expected argument of type {expected_type}, but got {type(arg)}")


def connect_to_milvus():
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)


def embed_text_with_openai(text):
    check_argument(text, str)
    try:
        embedding = embeddings.embed_documents(text)
        logging.info("Embedded text with OpenAI embeddings model.")
        return embedding
    except Exception as e:
        logging.error(
            f"Failed to embed text with OpenAI embeddings model: {e}")
        return None


def create_collection(name, schema):
    check_argument(name, str)
    check_argument(schema, CollectionSchema)
    client = connect_to_milvus()
    if client is not None:
        try:
            collection = Collection(name=name, schema=schema)
            logging.info(f"Created collection {name}.")
        except Exception as e:
            logging.error(f"Failed to create collection: {e}")


def store_vectors_with_milvus(vectors, vector_names, collection_name):
    check_argument(vectors, list)
    check_argument(vector_names, list)
    check_argument(collection_name, str)
    client = connect_to_milvus()
    if client is not None:
        try:
            collection = Collection(name=collection_name)
            collection.insert([vectors], ids=vector_names)
            logging.info(f"Stored vectors in collection {collection_name}.")
        except Exception as e:
            logging.error(f"Failed to store vectors: {e}")


def search_vectors_with_milvus(vectors, collection_name, param):
    check_argument(vectors, list)
    check_argument(collection_name, str)
    check_argument(param, dict)
    client = connect_to_milvus()
    if client is not None:
        try:
            collection = Collection(name=collection_name)
            result = collection.search(vectors, top_k=5, params=param)
            logging.info(f"Searched vectors in collection {collection_name}.")
            return result
        except Exception as e:
            logging.error(f"Failed to search vectors: {e}")
            return None


def get_user_input(prompt):
    check_argument(prompt, str)
    user_input = input(prompt)
    if not isinstance(user_input, str):
        logging.error("User input is not a string.")
        return None
    return user_input


def update_chapter_in_milvus(chapter_name, new_embedding, collection_name):
    check_argument(chapter_name, str)
    check_argument(new_embedding, list)
    check_argument(collection_name, str)
    if not isinstance(chapter_name, str) or not isinstance(collection_name, str):
        logging.error("Chapter name and collection name must be strings.")
        return
    client = connect_to_milvus()
    if client is not None:
        try:
            collection = Collection(name=collection_name)
            collection.delete(chapter_name)
            collection.insert([new_embedding], ids=[chapter_name])
            logging.info(f"Updated chapter {chapter_name} in collection "
                         f"{collection_name}.")
        except Exception as e:
            logging.error(f"Failed to update chapter: {e}")


def search_text_in_milvus(query, param):
    check_argument(query, str)
    check_argument(param, dict)
    if not isinstance(query, str):
        logging.error("Query must be a string.")
        return None
    query_embedding = embeddings.embed_query(query)
    results = search_vectors_with_milvus(
        [query_embedding], 'novel_chapters', param)
    return results


def generate_options_for_next_chapter(chapter, num_options):
    check_argument(chapter, str)
    check_argument(num_options, int)
    if not isinstance(chapter, str) or not isinstance(num_options, int):
        logging.error(
            "Chapter must be a string and num_options must be an integer.")
        return None
    results = search_text_in_milvus(chapter, {'nprobe': 16})
    if results is None:
        logging.error("Failed to generate options for the next chapter.")
        return None
    options = []
    for result in results:
        options.append(result[0].id)
    return options[:num_options]


def summerize_chapters(chapter):
    check_argument(chapter, str)
    if not isinstance(chapter, str):
        logging.error("Chapter must be a string.")
        return None
    results = search_text_in_milvus(chapter, {'nprobe': 16})
    if results is None:
        logging.error("Failed to summerize chapters.")
        return None
    return results[0][0].text


def get_next_chapter_from_user_input(chapter, num_options):
    check_argument(chapter, str)
    check_argument(num_options, int)
    if not isinstance(chapter, str) or not isinstance(num_options, int):
        logging.error(
            "Chapter must be a string and num_options must be an integer.")
        return None
    options = generate_options_for_next_chapter(chapter, num_options)
    if options is None:
        logging.error("Failed to get the next chapter from user input.")
        return None
    next_chapter = get_user_input(
        f"Select the next chapter from the following options: {options}")
    return next_chapter


def save_story_state(story_state, file_name):
    check_argument(story_state, dict)
    check_argument(file_name, str)
    with open(file_name, 'w') as f:
        json.dump(story_state, f)


def load_story_state(file_name):
    check_argument(file_name, str)
    # Load the story state from a file
    with open(file_name, 'r') as f:
        story_state = json.load(f)
    # Return the story state
    return story_state


def resume_story():
    # Load the story state from a file
    story_state = load_story_state('story_state.json')
    # Return the story state
    return story_state
