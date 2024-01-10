import socket
import logging
import os
import shutil
import subprocess
from typing import Any, Optional


import torch
from flask import Flask, jsonify, request
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings

# from langchain.embeddings import HuggingFaceEmbeddings
from run_localGPT import load_model
from prompt_template_utils import get_prompt_template

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from werkzeug.utils import secure_filename

import boto3

from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME

from dp_prompt_engine_settings import DpPromptEngineSettings

logging.basicConfig(level=logging.DEBUG)

if torch.backends.mps.is_available():
    DEVICE_TYPE = "mps"
elif torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
else:
    DEVICE_TYPE = "cpu"


# S3_SOURCES_BUCKET_NAME was "anl-dp-localgpt-source-documents
# S3_PREFIX was "Glue_Documentation/maurice_testdocs"
S3_SOURCES_BUCKET_NAME = "anl-dp-{env}-prompt-engine-artifacts"
S3_PREFIX = "SOURCE_DOCUMENTS"
LOCAL_SOURCES_FOLDER_NAME = "SOURCE_DOCUMENTS"

SHOW_SOURCES = True
logging.info(f"Running on: {DEVICE_TYPE}")
logging.info(f"Display Source Documents set to: {SHOW_SOURCES}")

EMBEDDINGS = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})

# uncomment the following line if you used HuggingFaceEmbeddings in the ingest.py
# EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
# if os.path.exists(PERSIST_DIRECTORY):
#     try:
#         shutil.rmtree(PERSIST_DIRECTORY)
#     except OSError as e:
#         print(f"Error: {e.filename} - {e.strerror}.")
# else:
#     print("The directory does not exist")

# run_langest_commands = ["python", "ingest.py"]
# if DEVICE_TYPE == "cpu":
#     run_langest_commands.append("--device_type")
#     run_langest_commands.append(DEVICE_TYPE)

# result = subprocess.run(run_langest_commands, capture_output=True)
# if result.returncode != 0:
#     raise FileNotFoundError(
#         "No files were found inside SOURCE_DOCUMENTS, please put a starter file inside before starting the API!"
#     )

# load the vectorstore
DB = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=EMBEDDINGS,
    client_settings=CHROMA_SETTINGS,
)


#MBI: zie https://github.com/langchain-ai/langchain/blob/fe7b40cb2a3628290d45de169498ccbcc73735d3/libs/langchain/langchain/schema/vectorstore.py#L562
RETRIEVER = DB.as_retriever(
    search_type="similarity_score_threshold",  # Added by Maurice :)
    search_kwargs={
        'score_threshold': DpPromptEngineSettings.SCORE_THRESHOLD, # score_threshold: Minimum relevance threshold for similarity_score_threshold
        'k': DpPromptEngineSettings.K_VALUE,  # Amount of documents to return (Default: 4)
    }     # Added by Maurice :)
)

LLM = load_model(device_type=DEVICE_TYPE, model_id=MODEL_ID, model_basename=MODEL_BASENAME)
prompt, memory = get_prompt_template(promptTemplate_type="llama", history=DpPromptEngineSettings.USE_HISTORY)
chain_type_kwargs = {"prompt": prompt}
if DpPromptEngineSettings.USE_HISTORY:
    chain_type_kwargs["memory"] = memory

QA = RetrievalQA.from_chain_type(
    llm=LLM,
    chain_type="stuff",
    retriever=RETRIEVER,
    return_source_documents=SHOW_SOURCES,
    chain_type_kwargs=chain_type_kwargs,
)

app = Flask(__name__)


@app.route("/api/delete_source", methods=["GET"])
def delete_source_route():
    folder_name = "SOURCE_DOCUMENTS"

    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)

    os.makedirs(folder_name)

    return jsonify({"message": f"Folder '{folder_name}' successfully deleted and recreated."})






class SyncS3ToSourceException(Exception):
    pass


def _download_dir(client: Any, resource: Any, prefix: str, local: str = "/tmp", bucket: str = "your_bucket") -> None:
    """
    Recursively download all files under a prefix from an S3 bucket to a local directory

    :param client: (boto3.client): Boto3 S3 client 
    :param resource (boto3.resource): Boto3 S3 resource
    :param prefix (str): S3 key prefix to download from 
    :param local (str): Local path to download files to, defaults to "/tmp"
    :param bucket (str): Name of S3 bucket, defaults to "your_bucket"
    """
    paginator = client.get_paginator("list_objects")
    for result in paginator.paginate(Bucket=bucket, Delimiter="/", Prefix=prefix):
        if result.get("CommonPrefixes") is not None:
            for subdir in result.get("CommonPrefixes"):
                _download_dir(client, resource, subdir.get("Prefix"), local, bucket)
        for file in result.get("Contents", []):
            dest_pathname = os.path.join(local, file.get("Key"))
            if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
            if not file.get("Key").endswith("/"):
                resource.meta.client.download_file(bucket, file.get("Key"), dest_pathname)


def get_account_id() -> Optional[str]:
    """Get ID of the AWS account that you are on.

    :return: the ID of the account. If this ID fails to be retrieved, None is
        returned instead.
    """
    # Get the AWS account ID so that we can determine if it is the dev or prd
    # environment.
    client = boto3.client("sts")
    caller_identity = client.get_caller_identity()
    if not caller_identity:
        print("Failed to retrieve caller identity.")
        return None

    return caller_identity["Account"]


ACCOUNT_MAP = {
    "297320915425": "sbox",
    "617760463975": "bidev",
    "658212563339": "biops", 
}

def get_account_name() -> Optional[str]:
    """Get name of the AWS environment that you are on. Returns None if the ID of the account fails to be retrieved.
    """
    account_nr = get_account_id()
    if account_nr is None:
        return None
    return ACCOUNT_MAP.get(account_nr, None)


def _insert_easter_eggs(local_path: str) -> None:
    """Insert easter eggs into the source documents."""
    file = os.path.join(local_path, "easter_egg_1.txt")
    with open(file, "w", encoding="utf-8") as f:
        f.write(
            "What is an epic? It's arguably one of humanity's greatest inventions. One notable user of the concept of Epic within Scrum and Safe is none other than Ron van Vliet. He's the famous RTE within a.s.r. that advocates the use of epics, Safe, Scrum, and karaoke"
        )
    file = os.path.join(local_path, "easter_egg_2.txt")
    with open(file, "w", encoding="utf-8") as f:
        f.write("What is glue? It's the stuff that no one wants to use, but on the other hand everyone needs. You can find it in AWS to do ETL and data analysis. But many people hate it. It's consdered useless and unfriendly. But after using it for a while, you'll find it's actually very useful and makes you intense happy.")


def copy_s3_to_local(bucket: str, prefix: str, local_path: str) -> None:
    """Copy objects from an S3 bucket to a local directory.
    
    :param bucket: Name of the S3 bucket.
    :param prefix: Only copy objects with keys starting with this prefix. 
    :param local_path: Local path to copy objects to.
    :return: None
    
    Examples:
        >>> copy_s3_to_local('mybucket', 'data/', '/tmp/data')

    This will copy all objects in the 'data/' prefix in 'mybucket' to the 
    '/tmp/data' directory locally.
    """
    client = boto3.client('s3')
    resource = boto3.resource('s3')
    _download_dir(client, resource, prefix=prefix, local=local_path, bucket=bucket)
    _insert_easter_eggs(local_path=local_path)

@app.route("/api/sync_s3_to_source_docs", methods=["GET"])
def sync_s3_to_source_docs_route():
    try:
        print("Remove all existing documents from the SOURCE_DOCUMENTS folder", end="... ")
        if os.path.exists(LOCAL_SOURCES_FOLDER_NAME):
            shutil.rmtree(LOCAL_SOURCES_FOLDER_NAME)
        os.makedirs(LOCAL_SOURCES_FOLDER_NAME)
        print("DONE")

        # Copy all the files on s3 recursively to the SOURCE_DOCUMENTS folder
        bucket_name = S3_SOURCES_BUCKET_NAME.format(env=get_account_name())

        print(f"Syncing files from {bucket_name}/{S3_PREFIX} to local {LOCAL_SOURCES_FOLDER_NAME}", end="... ")
        copy_s3_to_local(bucket=bucket_name, prefix=S3_PREFIX, local_path=LOCAL_SOURCES_FOLDER_NAME)

        return f"Files synched successfully from {bucket_name}/{S3_PREFIX} to local {LOCAL_SOURCES_FOLDER_NAME}", 200

    except Exception as e:
        print(f"Error during sync_s3_to_source_docs_route: {e}.")
        return f"Error occurred: {str(e)}", 500


@app.route("/api/save_document", methods=["GET", "POST"])
def save_document_route():
    if "document" not in request.files:
        return "No document part", 400
    file = request.files["document"]
    if file.filename == "":
        return "No selected file", 400
    if file:
        filename = secure_filename(file.filename)
        folder_path = "SOURCE_DOCUMENTS"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, filename)
        file.save(file_path)
        return "File saved successfully", 200


@app.route("/api/run_ingest", methods=["GET"])
def run_ingest_route():
    global DB
    global RETRIEVER
    global QA
    try:
        if os.path.exists(PERSIST_DIRECTORY):
            try:
                shutil.rmtree(PERSIST_DIRECTORY)
            except OSError as e:
                print(f"Error: {e.filename} - {e.strerror}.")
        else:
            print("The directory does not exist")

        run_langest_commands = ["python", "ingest.py"]
        if DEVICE_TYPE == "cpu":
            run_langest_commands.append("--device_type")
            run_langest_commands.append(DEVICE_TYPE)

        result = subprocess.run(run_langest_commands, capture_output=True)
        if result.returncode != 0:
            return "Script execution failed: {}".format(result.stderr.decode("utf-8")), 500
        # load the vectorstore
        DB = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=EMBEDDINGS,
            client_settings=CHROMA_SETTINGS,
        )
        RETRIEVER = DB.as_retriever()
        prompt, memory = get_prompt_template(promptTemplate_type="llama", history=DpPromptEngineSettings.USE_HISTORY)

        QA = RetrievalQA.from_chain_type(
            llm=LLM,
            chain_type="stuff",
            retriever=RETRIEVER,
            return_source_documents=SHOW_SOURCES,
            chain_type_kwargs={
                "prompt": prompt,
            },
        )
        return "Script executed successfully: {}".format(result.stdout.decode("utf-8")), 200
    except Exception as e:
        return f"Error occurred: {str(e)}", 500


@app.route("/api/prompt_route", methods=["GET", "POST"])
def prompt_route():
    global QA
    user_prompt = request.form.get("user_prompt")
    if user_prompt:
        # print(f'User Prompt: {user_prompt}')
        # Get the answer from the chain
        res = QA(user_prompt)
        answer, docs = res["result"], res["source_documents"]

        prompt_response_dict = {
            "Prompt": user_prompt,
            "Answer": answer,
        }

        prompt_response_dict["Sources"] = []
        for document in docs:
            prompt_response_dict["Sources"].append(
                (os.path.basename(str(document.metadata["source"])), str(document.page_content))
            )
            print(document)
            logging.info(f"Running on: {document}")
            print(document.metadata)
            logging.info(f"Running on: {document.metadata}")

        return jsonify(prompt_response_dict), 200
    else:
        return "No user prompt received", 400


@app.route("/api/get_scores", methods=["GET", "POST"])
def get_scores():
    global DB
    user_prompt = request.form.get("user_prompt")
    if user_prompt:
        # Find the relevant pages
        search = DB.similarity_search_with_score(user_prompt)  # returns List[Tuple[Document, float]]

        doc_list = []
        for doc in search:
            location = doc[0].metadata['source'].split('/')[-1]
            distance = doc[1]
            doc_list.append((location, distance))
        return jsonify(doc_list), 200
    else:
        return "No user prompt received", 400


def get_ip_address() -> str:
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    app.run(debug=False, port=5110)
