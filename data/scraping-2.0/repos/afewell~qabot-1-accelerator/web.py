import json
import os
import openai
import gradio as gr
import logging
import sys
from langchain.chat_models import ChatOpenAI
from llama_index import (
    LLMPredictor,
    ServiceContext,
    StorageContext,
    load_index_from_storage
)

# import/from statements omitted for brevity
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))  
os.environ['OPENAI_API_KEY'] = 'FAKE_PLACEHOLDER_TO_INITIALIZE_INDEX'
openai.api_key = os.environ.get('OPENAI_API_KEY')
# The llm predictor is initialized with a placeholder fake api key and model selection, this can be replaced by user input at runtime without needing to reload the index
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size=512)



def construct_index():
    index = None
    try:
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
    except:
        print("the storage context did not load correctly")
        exit
    return index

def chatbot(input_text, api_key, model_name):
    global service_context
    global openai
    os.environ['OPENAI_API_KEY'] = api_key
    openai.api_key = api_key
    index.service_context.llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name=model_name, streaming=True, openai_api_key=api_key))
    query_engine = index.as_query_engine()
    response = query_engine.query(input_text)
    print("Response:", response.response)
    try:
        json.loads(response.response)
    except json.JSONDecodeError as e:
        print("Invalid JSON:", e)
    return response.response

model_list = ['gpt-3.5-turbo', 'gpt-4']  # Update this with your list of models

index = construct_index()
iface = gr.Interface(
    fn=chatbot,
    inputs=[
        gr.inputs.Textbox(lines=7, label="How may I help you?"),
        gr.inputs.Textbox(label="Your OpenAI API Key"),
        gr.inputs.Dropdown(choices=model_list, label="Model Name"),
    ],
    outputs="text",
    title="Tanzu-Trained-ChatBot",
)

iface.launch(debug=True, server_name="0.0.0.0")