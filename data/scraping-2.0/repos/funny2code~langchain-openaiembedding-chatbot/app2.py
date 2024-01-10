from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper, StorageContext, load_index_from_storage
from langchain.chat_models import ChatOpenAI 
import gradio as gr 
import sys
import os

os.environ["OPENAI_API_KEY"] = "your_openai_key"

def construct_index(directory_path): 
    max_input_size = 4896
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 681 
    prompt_helper = PromptHelper(max_input_size, num_outputs, chunk_overlap_ratio= 0.1, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-4", max_tokens=num_outputs))
    documents = SimpleDirectoryReader(directory_path).load_data()
    index = GPTVectorStoreIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    # index.save_to_disk('index.json')
    index.storage_context.persist('./')
    return index

def chatbot(input_text): 
    storage_context = StorageContext.from_defaults(persist_dir="./")
    # index = GPTVectorStoreIndex.load_from_disk('index.json')
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    response = query_engine.query(input_text)
    return response.response

iface = gr.Interface(fn=chatbot, 
                    inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                    outputs="text",
                    title="TheoloGPT")

index = construct_index("docs")
iface.launch(share=True)
