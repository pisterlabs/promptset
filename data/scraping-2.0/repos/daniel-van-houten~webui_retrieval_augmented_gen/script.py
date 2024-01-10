import gradio as gr
import torch
import chromadb
from modules import chat, shared
from langchain.embeddings import HuggingFaceBgeEmbeddings
from .prompts import QA
from langchain.vectorstores import Chroma

# Constants
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global variables
CHROMA_CLIENT = None
source_documents = []
params = {
    "enabled": True,
    "k": 3,
    "relevance_threshold": 0.5,
    "host": "localhost",
    "port": 8000,
    "collection_name": "custom_data"
}

embedding_function = HuggingFaceBgeEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": DEVICE},
            encode_kwargs = {'normalize_embeddings': True})

class ChromaClient:
    def __init__(self, params):
        self.params = params

    def load_chroma_client(self):
        global CHROMA_CLIENT

        try:
            chromadb_client = chromadb.HttpClient(host=self.params["host"], port=self.params["port"])
            chroma_collection = chromadb_client.get_collection(self.params["collection_name"])
            total_docs = chroma_collection.count()
            CHROMA_CLIENT = Chroma(client=chromadb_client, collection_name=self.params["collection_name"], embedding_function=embedding_function)
        except Exception as e:
            shared.logger.error(f"Failed to connect to Chroma DB: {e}")
            return None, 0

        return CHROMA_CLIENT, total_docs


def load_chroma_client(params):
    global CHROMA_CLIENT

    chromadb_client = chromadb.HttpClient(host=params["host"], port=params["port"])

    chroma_collection = chromadb_client.get_collection(params["collection_name"])
    total_docs = chroma_collection.count()

    CHROMA_CLIENT = Chroma(client=chromadb_client, collection_name=params["collection_name"], embedding_function=embedding_function)

    return CHROMA_CLIENT, total_docs

def history_modifier(history):
    """
    Modifies the chat history.
    Only used in chat mode.
    """
    return history

def state_modifier(state):
    """
    Modifies the state variable, which is a dictionary containing the input
    values in the UI like sliders and checkboxes.
    """
    return state

def chat_input_modifier(text, visible_text, state):


    """
    Modifies the user input string in chat mode (visible_text).
    You can also modify the internal representation of the user
    input (text) to change how it will appear in the prompt.
    """
    return text, visible_text

def output_modifier(string, is_chat=False):
    """
    Modifies the LLM output before it gets presented.

    In chat mode, the modified version goes into history['visible'],
    and the original version goes into history['internal'].
    """
    global source_documents

    if not params['enabled'] or CHROMA_CLIENT is None:
        return string

    string = string + "\n---\n" + "\n".join([ f" - **Source**: {source} | **Score**: {round(score, 2)}" for source, score in source_documents]) 

    return string

def input_modifier(string):
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """

    global CHROMA_CLIENT
    global source_documents
    
    source_documents = []

    # If collection could not be loaded, just return the input
    if not params['enabled'] or CHROMA_CLIENT is None:
        return string

    user_input = string

    shared.logger.info(f"Searching for contextual docs using input '{user_input}'" )
    query_results = CHROMA_CLIENT.similarity_search_with_relevance_scores(user_input, k=params['k'], kwargs={"score_threshold": params['relevance_threshold']})

     # if there are no contextual docs, just return the string
    if len(query_results) == 0:
        shared.logger.info("No related docs returned. Just defaulting to original user input")
        return string

    source_documents = [(result.metadata['source'], score) for result, score in query_results]

    context_str = "\n\n".join([ f"Contextual Score: {score}\n{result.page_content}" for result, score in query_results])

    prompt = QA.format(context=context_str, instruction=user_input)

    return prompt

def bot_prefix_modifier(string, state):
    """
    Modifies the prefix for the next bot reply in chat mode.
    By default, the prefix will be something like "Bot Name:".
    """
    return string

def custom_generate_chat_prompt(user_input, state, **kwargs):
    """
    Replaces the function that generates the prompt from the chat history.
    Only used in chat mode.
    """
    result = chat.generate_chat_prompt(user_input, state, **kwargs)
    return result

def custom_css():
    """
    Returns a CSS string that gets appended to the CSS for the webui.
    """
    return ''

def custom_js():
    """
    Returns a javascript string that gets appended to the javascript
    for the webui.
    """
    return ''

def setup():
    """
    Gets executed only once, when the extension is imported.
    """
    pass

def ui():
    global total_docs

    chroma_client = ChromaClient(params)

    with gr.Accordion("Settings"):
        status = gr.Textbox("Disconnected: Connect to Chroma DB", label="status")

        with gr.Row():
            host = gr.Textbox(value=params['host'], label='Host')
            port = gr.Textbox(value=params['port'], label='Port')
            collection_name = gr.Textbox(value=params['collection_name'], label='Collection Name')

        with gr.Row():
            connect_button = gr.Button("Connect to Chroma DB")

        with gr.Row():
            enabled = gr.Checkbox(value=params['enabled'], label='Enabled')
    
        with gr.Row():
            k_docs = gr.Slider(value=params['k'], label='k docs', minimum=1, maximum=10, step=1)
        
        with gr.Row():
            relevance_threshold = gr.Slider(value=params['relevance_threshold'], label='Document Relevance Threshold', minimum=0.05, maximum=1, step=0.05)

    # Update params on UI change
    host.change(lambda x: params.update({'host': x}), host, None)
    port.change(lambda x: params.update({'port': x}), port, None)
    collection_name.change(lambda x: params.update({'collection_name': x}), collection_name, None)
    enabled.change(lambda x: params.update({'enabled': x}), enabled, None)
    k_docs.change(lambda x: params.update({'k': x}), k_docs, None)
    relevance_threshold.change(lambda x: params.update({'relevance_threshold': x}), relevance_threshold, None)

    # Event handlers
    connect_button.click(fn=lambda: f"Connected: {chroma_client.load_chroma_client()[1]} Document Embdeddings", outputs=[status])