import streamlit as st
from PIL import Image
import os
import openai
from utils.redis_helpers import connect_redis, reformat_redis, upload_to_redis, create_query_context
from utils.vicuna_helpers import clear_history, query, get_http_response_text, extract_answer
from utils.jarvis_helpers import run_alexa
from utils.ocr import ocr_files, get_db_schema
from streamlit_chat import message
import time

st.set_page_config(
    page_title="Upload Docs",
    page_icon="📝",
)

st.write("# Data Driven Responses ")

st.sidebar.success("Upload documents and parse them easily with Jarvis!")

st.header("Upload your documents data to JarvIEs")
with st.sidebar:
    embeddings_preference = st.selectbox("Choose your embeddings preference", ["speed", "more_speed", "quality"], index=0)

# ------ CONSTANTS ------
redis_conn = connect_redis()
controller_addr = "http://localhost:21001"
worker_addr = "http://localhost:21002"



tempDir = "tempDir" # TBD - Delete tempDir after processing

files = st.file_uploader("Upload & process your data", type=["pdf"], accept_multiple_files=True)


if not os.path.exists(tempDir):
    os.makedirs(tempDir)   

left, right = st.columns([4,1])
if left.button("Process data"):     
    if files is not None:
        file_names = []    
        dfs = []
        print(files)
        for file in files:
            st.write(file)
            # Open and save file to temp folder
            with open(os.path.join(tempDir, file.name), "wb") as f:
                f.write(file.getbuffer())
            
            st.write("Processing data")
            sentences = ocr_files(tempDir)
            db_schema = get_db_schema(sentences, embeddings_preference)
            #print("Successfully processed")
            #print("Uploading data to Redis")
            for key, value in db_schema.items():
                upload_to_redis(
                    value[0], 
                    value[1],
                    value[2],
                    value[3],
                    value[4],
                    redis_conn)
            print("Index size: ", redis_conn.ft().info()['num_docs'])
        st.write("Successfully uploaded data to your Redis database")    
    else:
        st.write("No files uploaded!")
if right.button("Restart Redis"):
    try:
        redis_conn = connect_redis()
        reformat_redis(redis_conn, embeddings_preference)
        st.write("Successfully formatted Redis")
    except:
        st.error("RedisError: Please check the connection with the Redis server!")
