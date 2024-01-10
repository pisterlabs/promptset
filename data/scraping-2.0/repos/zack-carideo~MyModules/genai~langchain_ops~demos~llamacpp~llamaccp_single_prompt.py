"""
demo 
"""
import os
from huggingface_hub import snapshot_download, hf_hub_download 
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate

#model informatoin and download settings
repo_id = "shaowenchen/llama-2-7b-langchain-chat-gguf"
filename = 'llama-2-7b-langchain-chat.Q4_K.gguf'
repo_type = "model"
local_dir = "/home/zjc1002/Mounts/llms/llama-2-7b-langchain-chat-gguf"
local_dir_use_symlinks = False
model_path = Path(local_dir, filename) 

#model memory settings
n_gpu_layers = 4 # Change this value based on your model and your GPU VRAM pool.
n_batch = 10    # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

#prompt template
#WARNING: prompt templates are model specific, make sure to check the models pages on huggingface.co for the correct prompt template
template = """Question: {question}
Answer: Let's work this out in a step by step way to be sure we have the right answer."""

#define the question to ask the llm
question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

# Check if local_dir exists
if not os.path.exists(local_dir):
    os.makedirs(local_dir)

    #download a model 
    hf_hub_download(
        repo_id = repo_id
        , filename = filename
        , repo_type = repo_type
        , local_dir = local_dir
        , local_dir_use_symlinks = local_dir_use_symlinks
        )

else:
    print("local_dir already exists, skipping download")


#define prompt template
prompt = PromptTemplate(
        template=template
        , input_variables=["question"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
from pathlib import Path 
llm = LlamaCpp(
    model_path=model_path
    , n_gpu_layers=n_gpu_layers
    , n_batch=n_batch
    , callback_manager=callback_manager
    , verbose=True,  # Verbose is required to pass to the callback manager
)

# Run the chain with the prompt and llm
llm_chain = LLMChain(prompt=prompt, llm=llm)
llm_chain.run(question)

#NEXT STEPS 
#1. use grammers to constrain model outputs and sample tokens based on the rules defined in the .gbnf file