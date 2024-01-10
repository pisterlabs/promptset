from llama_index import GPTVectorStoreIndex, load_index_from_storage, StorageContext
import os
import gradio as gr
import openai

os.environ['OPENAI_API_KEY'] = "<API-KEY>"

def chatbot(input_text):
    openai.api_key = os.environ['OPENAI_API_KEY']
    storage_context = StorageContext.from_defaults(persist_dir='Store')
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    response = query_engine.query(input_text)
    return response.response


app = gr.Interface(fn=chatbot,
                   inputs=gr.inputs.Textbox(lines=5,label="Send a message"),
                   outputs="text",
                   title="Tekhmos Chatbot")

app.launch(share=False)