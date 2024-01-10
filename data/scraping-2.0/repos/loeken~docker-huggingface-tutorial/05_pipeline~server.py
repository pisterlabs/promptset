# https://github.com/amrrs/LLM-QA-Bot/blob/main/LLM_Q%26A_with_Open_Source_Hugging_Face_Models.ipynb
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex,GPTSimpleVectorIndex, PromptHelper
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LLMPredictor, ServiceContext
import torch
from langchain.llms.base import LLM
from transformers import pipeline

class customLLM(LLM):
    model_name = "google/flan-t5-large"
    pipeline = pipeline("text2text-generation", model=model_name, device=0, model_kwargs={"torch_dtype":torch.bfloat16})

    def _call(self, prompt, stop=None):
        return self.pipeline(prompt, max_length=9999)[0]["generated_text"]

    def _identifying_params(self):
        return {"name_of_model": self.model_name}

    def _llm_type(self):
        return "custom"


llm_predictor = LLMPredictor(llm=customLLM())

hfemb = HuggingFaceEmbeddings()
embed_model = LangchainEmbedding(hfemb)

text1 = """TEST STRING """

#documents = SimpleDirectoryReader('data').load_data()

from llama_index import Document

text_list = [text1]

documents = [Document(t) for t in text_list]

# # set number of output tokens
# num_output = 500
# # set maximum input size
# max_input_size = 512
# # set maximum chunk overlap
# max_chunk_overlap = 15


# prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    

#index = GPTSimpleVectorIndex(documents, embed_model=embed_model, llm_predictor=llm_predictor)

#index = GPTListIndex(documents, embed_model=embed_model, llm_predictor=llm_predictor)

#index.save_to_disk('index.json')

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)
index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)



import logging

logging.getLogger().setLevel(logging.CRITICAL)
response = index.query( "What's the cost of Whisper model?") 

print(response)

# temp fix for running shell commands on Google Colab

import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding
import gradio as gr

index = None

def build_the_bot(input_text):
    text_list = [input_text]
    documents = [Document(t) for t in text_list]
    global index
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
    return('Index saved successfull!!!')