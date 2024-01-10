# Internal files
import mongo_db_connector as mongo
import spaces_connector as spaces
import config

# External files
from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, Document
from langchain.chat_models import ChatOpenAI
# import gradio as gr
import os
import time
from datetime import datetime

# -----------------------------------------------------------------------------------------------------------------------------
# Function to build the index for OpenAI ChatBot. 
# Prereqs: 
#       - config.py containts the correct OpenAI api key; config.api_key
#       - config.py is configured with the connection string to the mongo DB; variable name -> mongo_string
#       - config.py is has the correct database to search to the mongo DB; variable name -> MONGO_DATABASE
#       - config.py has spaces configurations OBJECT_STORAGE_KEY, OBJECT_STORAGE_SECRET, OBJECT_STORAGE_REGION, and OBJECT_STORAGE_BUCKET
# -----------------------------------------------------------------------------------------------------------------------------

os.environ["OPENAI_API_KEY"] = config.api_key

SPACES_JSON_FILE_NAME = "initial_index1.json"

def build_index(documents):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('./index.json')

    return index

org_list = ["566f6980-2166-4718-ba88-77610e998cbd"]

json_list = []
for item in mongo.get_refined(org_list):
    # Some dictionaries do not have key word counts. If they do not, do not append them to the string
    if "keywords" in item:
        new_string = str(item["page_url"]) + " " + str(item["summary"]) + str(item["keywords"]["keyword_counts"])
    else:
        new_string = str(item["page_url"]) + " " + str(item["summary"])   
    json_list.append(new_string)

documents = [Document(t) for t in json_list]

start_time = time.time()
index = build_index(documents=documents)
end_time = time.time()
completion_time = end_time-start_time
print(f"index build time: ", completion_time)




#Configure s3 connection
s3config = {
    "region_name": config.OBJECT_STORAGE_REGION,
    "endpoint_url": "https://{}.digitaloceanspaces.com".format(config.OBJECT_STORAGE_REGION),
    "aws_access_key_id": config.OBJECT_STORAGE_KEY,
    "aws_secret_access_key": config.OBJECT_STORAGE_SECRET,
    "bucket_name": config.OBJECT_STORAGE_BUCKET}

today = str(datetime.today())
# Set up the metadata to attach to the spaces storage
metadata = {'Orgids': ','.join(org_list), 'Ingestion_Date': today}

# Upload the indexed file to spaces for storage. 
spaces.upload_file_spaces(s3config["bucket_name"], 
    './index.json', SPACES_JSON_FILE_NAME, 
    s3config["endpoint_url"], s3config["aws_access_key_id"], 
    s3config["aws_secret_access_key"], metadata = metadata)
