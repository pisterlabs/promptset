import streamlit as st
from PIL import Image
import os
import openai
from utils.redis_helpers import connect_redis, reformat_redis, upload_to_redis, create_query_context
from utils.ocr import ocr_files, get_db_schema
from streamlit_chat import message
from utils.common import load_models , LOG, env
from utils.ocr import ocr_files, get_db_schema
from streamlit_chat import message
from utils.model_inference_helpers import query_model, query_openai
from openai.error import OpenAIError


LOG.info(f"Environment variables loaded: {env.controller} {env.worker}")
models = load_models()

# # Redis connection details
# endpoint = "127.0.0.1"
# port = 6379
# username = "default"

#redis_conn = connect_redis()
controller_addr = env.worker
worker_addr = env.controller
LOG.info(f"Redis connection: {controller_addr} {worker_addr}")



st.set_page_config(
    page_title="Upload Docs",
    page_icon="📝",
)

with st.sidebar:
    model_select = st.radio(
        "Elija el modelo que desea usar",
        ('MarIA Base', 'MarIA large', 'Beto Base Spanish Sqac', 'mrm8488', 'ChatGPT'))
    
    model_preference_hf_url = models[model_select] # full url to model 
    LOG.info(f" Model selected: {model_preference_hf_url}")
    st.write(f"Modelo seleccionado: \n\n {models[model_select]}")

    if model_preference_hf_url == 'openai': 
        gpt_engine = st.selectbox("Model Name",
                                  ["text-davinci-003", "text-davinci-002", "davinci"])
        max_new_tokens = st.number_input("Max Tokens", value=500, step=100)
        
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

st.header(f'Habla con {model_select} ')

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
    # TODO: Change this to a function
    # TODO: Add a progress bar
         
    if files is not None:
        file_names = []    
        dfs = []
        LOG.info("Processing data")
        for file in files:
            st.write(file)
            # Open and save file to temp folder
            with open(os.path.join(tempDir, file.name), "wb") as f:
                f.write(file.getbuffer())
            
            st.write("Processing data")
            sentences = ocr_files(tempDir) # takes the sentences from the pdfs
            LOG.info(f"Sentences from ocr_files : {sentences}")
            db_schema = get_db_schema(sentences) 
            for key, value in db_schema.items():
                upload_to_redis(
                    value[0], 
                    value[1],
                    value[2],
                    value[3],
                    value[4],
                    redis_conn)
            print(redis_conn.ft().info()['num_docs'])
        st.write("Successfully uploaded data to your Redis database")    
    else:
        st.write("No files uploaded!")
if right.button("Restart Redis"):
    try:
        redis_conn = connect_redis()
        LOG.info(redis_conn)
        reformat_redis(redis_conn)
        st.write("Successfully formatted Redis")
        #remove files from tempDir: TODO Create a ui to show which files are present in tempDir and remove them through the UI
        for file in os.listdir(tempDir):
            os.remove(os.path.join(tempDir, file))
        st.write("Successfully deleted files from tempDir")
    except:
        st.error("RedisError: Please check the connection with the Redis server!")


user_query = st.text_area('Send your message', '')
left, right = st.columns([3,1])

if right.button("Clear Chat History", key="clear"):
    st.session_state['generated'] = []
    st.session_state['past'] = []

if left.button("Submit"):
    st.session_state['past'].append(user_query)
   
    if model_preference_hf_url != "openai":
        assistant_prompt = create_query_context(redis_conn, user_query, model=model_preference_hf_url)
        with st.expander("See generated prompt"):
            st.text(assistant_prompt)
        try:
            res = query_model(model_preference_hf_url, assistant_prompt)
            st.session_state['generated'].append(res['answer'])
        except:
            st.warning("Something went wrong!")
            output = "None"
    else:
        assistant_prompt = create_query_context(redis_conn, user_query, model=model_preference_hf_url)
        with st.expander("See generated prompt"):
            st.text(assistant_prompt)
        try: 
            res = query_openai(gpt_engine, assistant_prompt, max_new_tokens)
            
        except OpenAIError as e:
            LOG.info(f"OpenAIError: {e}")
            st.warning("Something went wrong!")
            res = e
        
        st.session_state['generated'].append(res)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')