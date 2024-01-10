# Have two models converse with each other

import os
import utils as u

# not used on mac
os.environ['CUBLAS_LOGINFO_DBG'] = '1'

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
import httpx
from llama_cpp.llama import Llama, LlamaGrammar


# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# ----------------------------------------------------------------------

n_gpu_layers = 20  # Change this value based on your model and your GPU VRAM pool.
n_batch = 128  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

LLM_MODELS = os.environ['LLM_MODELS']

modelA = LLM_MODELS+"zephyr-7b-beta.Q4_K_M.gguf"
#modelB = LLM_MODELS+"samantha-mistral-instruct-7b.Q4_K_M.gguf"
modelB = LLM_MODELS+"mistral-7b-instruct-v0.1.Q3_K_M.gguf"

# Make sure the model path is correct for your system!
def myLlamaCpp(model):
    llm = LlamaCpp(
        model_path=model,
        n_gpu_layers=n_gpu_layers,
        n_ctx=2048,
        stop=[],
        max_tokens=1000,
        n_threads=8,
        temperature=0.4,  # also works with 0.0 (0.01 is safer)
        f16_kv=True,
        n_batch=n_batch,
        callback_manager=callback_manager,
        verbose=False,  
    )
    return llm

llmA = myLlamaCpp(modelA)
llmB = myLlamaCpp(modelB)

contextA = "You are Sir Isaac Newton, believing the idea of an absolute space background and the concept of absolute time. Please give your thoughts on a quantum  theory of gravity."
contextB = "You are Albert Einstein, who developed special and general relativity where everything is relative, and even time depends on relative position and speed. Please give your thoughts on a quantum  theory of gravity."

print(dir(llmA))

print("=====================")
llmA(contextA)
print("=====================")
llmB(contextB)
print("=====================")
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
llmA(contextA)
print("=====================")
llmB(contextB)
# ----------------------------------------------------------------------
# How would I like to use this system and track the conversation? 
# ----------------------------------------------------------------------
