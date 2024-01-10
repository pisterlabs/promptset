from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain import OpenAI
from IPython.display import Markdown, display
import sys
import os
import glob
import pandas as pd
import gradio as gr
from flask import Flask

os.environ["OPENAI_API_KEY"] = "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"
app = Flask(__name__)


def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    # define prompt helper
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="ada-search-document", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

    index.save_to_disk('index.json')
    return index


def ask_ai(query):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(query)
    return response.response


# Specify the directory path containing your documents
directory_path = 'data'

# Construct the index
index = construct_index(directory_path)


iface = gr.Interface(
    fn=ask_ai,
    inputs="text",
    outputs="text",
    title="AI Chat",
    description="Ask any question and get an AI-generated response."
)

@app.route('/')
def home():
    return iface.launch()


if __name__ == "__main__":
    iface.launch(share=True)
    app.run()