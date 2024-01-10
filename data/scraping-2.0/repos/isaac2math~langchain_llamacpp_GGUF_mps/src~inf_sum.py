import os
import platform
import traceback

from pathlib import Path

from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import LlamaCpp
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackManager

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

#chdir
if platform.system() == 'Darwin':

    os.chdir("/Users/ning/Library/CloudStorage/Dropbox/Working_Directory/NLP/langchain_llama2/")
    print("the current wd is ", os.getcwd())

elif platform.system() == 'Linux':
    
    print("this branch is developed for mps,")
    print("please use the GGUF_linux branch")

else :
    
    print("I hate Windows")


llm = LlamaCpp(model_path="./model/llama-2-70b.Q5_K_M.gguf", 
               n_threads=12,
               n_parts=-1,
               n_gpu_layers=1, 
               n_batch=512, 
               callback_manager=callback_manager, 
               verbose=True,
               max_tokens=10000, 
               n_ctx=8192)

text_splitter = CharacterTextSplitter()

with open("./data/feature/test1.txt") as f:
    raw_text = f.read()
texts = text_splitter.split_text(raw_text)

docs = [Document(page_content=t) for t in texts[:3]]

chain = load_summarize_chain(llm, chain_type="refine")
chain.run(docs)

