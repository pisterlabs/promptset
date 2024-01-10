import ssl

import nltk
from langchain.document_loaders import BSHTMLLoader, DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import ElasticsearchStore
import os
import boto3
import time as t

from config import Paths, openai_api_key


def download_files_from_s3(bucket_name, directory, local_directory, batch_size=50):
    s3 = boto3.client('s3')

    # Create the local directory if it doesn't exist
    os.makedirs(local_directory, exist_ok=True)

    # Initialize ContinuationToken to None
    continuation_token = None

    while True:
        # List objects with continuation token
        list_objects_params = {'Bucket': bucket_name, 'Prefix': directory}
        if continuation_token:
            list_objects_params['ContinuationToken'] = continuation_token

        objects = s3.list_objects_v2(**list_objects_params)

        # Download files in batches
        for i in range(0, len(objects['Contents']), batch_size):
            batch_objects = objects['Contents'][i:i+batch_size]
            if continuation_token is None:
                batch_objects.pop(0)

            # Download each file in the batch
            for obj in batch_objects:
                key = obj['Key']
                local_path = os.path.join(local_directory, os.path.basename(key))

                # Download the file
                s3.download_file(bucket_name, key, local_path)
                print(f"Downloaded: {key} -> {local_path}")

            # Execute function_x on the batch of downloaded files
            load_data()
            delete_downloaded_files(local_directory)

        # Check if there are more objects to retrieve
        if not objects.get('NextContinuationToken'):
            break

        # Set the continuation token for the next iteration
        continuation_token = objects['NextContinuationToken']

def delete_downloaded_files(directory):
    # Delete all files in the specified directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

def load_data():
    repo_url = "https://github.com/AndresMarcelo7/TwitterAYGO"
    markdown_path = "../src/data/markdown"
    loader = DirectoryLoader(markdown_path, glob="**/*.md")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    documents = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    db = ElasticsearchStore.from_documents(
        documents,
        es_cloud_id=os.getenv("ELASTIC_CLOUD_ID"),
        index_name="elastic-index",
        embedding=embeddings,
        es_api_key=os.getenv("ELASTIC_API_KEY")
    )
    print(db.client.info())


def installnltk():

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")


if __name__ == "__main__":
    #installnltk()
    start = t.time()
    download_files_from_s3(os.getenv("S3_BUCKET_NAME"),os.getenv("S3_DATA_PREFIX"),"data/markdown")
    end = t.time()
    total = end - start
    print("Total Execution time indexer - Elasticsearch: " + str(total))

