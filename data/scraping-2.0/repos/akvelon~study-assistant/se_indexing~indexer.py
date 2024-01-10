"""DB INDEXER"""
import os
import pickle
import json
import glob
import multiprocessing
import openai
from db_engine.config import get_database
from settings import settings


def get_documents():
    """Find all documents from the documents/ folder and return absolute path."""

    documents_folder = os.path.join(os.getcwd(), "se_indexing/documents/")

    # Find all files in documents/ folder
    document_files = glob.glob(os.path.join(documents_folder, "*.json"), recursive=True)

    for document in document_files:
        yield document

    # return document_files


# TODO: Implement deleting a document when finished indexing


def create_summary_for_content(content):
    """Creates summary using content for document"""
    max_tokens = 4097
    summary_word_count = 100  # words
    tokens_per_word = 100 / 75  # tokens per word
    summary_token_count = int(summary_word_count * tokens_per_word)  # words
    messages = [
        # Prepare ChatGPT for summarizing
        {
            "role": "system",
            # pylint: disable=line-too-long
            "content": """Summarize the following content as concisely as possible. Max word count is 400.""",
        },
        {
            "role": "user",
            "content": content[: max_tokens - summary_token_count],
        },
    ]

    gpt_response = openai.ChatCompletion.create(
        model=settings.chatgpt_model,
        messages=messages,
    )
    summary = gpt_response.choices[0].message.content

    return summary


def create_embedding_for_summary(summary):
    """Creates embedding for summary"""
    embedding_reply = openai.Embedding.create(
        model=settings.embedding_model,
        input=summary,
    )
    embedding_list = embedding_reply["data"][0]["embedding"]

    # Serialize embedding_list into a bytestream
    embedding = pickle.dumps(embedding_list, pickle.HIGHEST_PROTOCOL)

    return embedding


def process_document(document_abspath):
    """Document indexing func"""
    database = get_database()

    with open(document_abspath, "r", encoding="utf-8") as document_file:
        print(f"Processing {document_file.name}")
        document = json.load(document_file)
        document_content = document["content"]

        # Check if document is already in database
        if database.find_document_by_url(document["url"]) is None:
            document_summary = create_summary_for_content(document_content)
            document_embedding = create_embedding_for_summary(document_summary)

            document_id = database.insert_document(document)
            summary_id = database.insert_summary(document_id, document_summary)
            database.insert_embedding(
                summary_id, settings.embedding_model, document_embedding
            )

    # TODO: Delete document from documents/ folder
    # delete_document(document_filename)


def main():
    """Main func"""
    with get_database() as database:
        database.create_database_if_not_exists()

        openai.api_key = settings.openai_key

        with multiprocessing.Pool(processes=8) as pool:
            for document_abspath in get_documents():
                document_filename = os.path.basename(document_abspath)
                print(f"Indexing {document_filename}")

                # this is so awesome
                pool.apply_async(process_document, args=(document_abspath,))
            pool.close()
            pool.join()


main()
