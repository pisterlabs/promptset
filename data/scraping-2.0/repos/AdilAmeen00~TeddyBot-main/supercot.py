#--- supercot  -> https://huggingface.co/ausboss/llama-30b-supercot
import sys
sys.argv = [sys.argv[0]]
import os
import re
import time
import json
from pathlib import Path
import transformers
from transformers import pipeline
# from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
sys.path.append(str(Path().resolve().parent / "modules"))
# from modules import chat, shared, training, ui, utils
# from modules.html_generator import chat_html_wrapper
# from modules.LoRA import add_lora_to_model
# from modules.models import load_model, load_soft_prompt
# from modules.text_generation import generate_reply, stop_everything_event
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel  
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse
import torch
#  -------------------------

# Code for Supercot model  ---------------
tokenizer = AutoTokenizer.from_pretrained("ausboss/llama-30b-supercot")
model = AutoModelForCausalLM.from_pretrained("ausboss/llama-30b-supercot")

pipe = pipeline(
    "text-generation",
    model=model, 
    tokenizer=tokenizer,
    device=0,
    max_length=1200,
    temperature=0.6,
    top_p=0.95,
    repetition_penalty=1.1
)
llm = HuggingFacePipeline(pipeline=pipe)

# To save the embeddings model
save_directory = "C:\Code\PrivateGPT\privateGPT-main\model_super"
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

llm = LlamaCpp(model_path=save_directory)