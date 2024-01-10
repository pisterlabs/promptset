import os
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from llama_cpp.llama import Llama, LlamaGrammar

template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
#----------------------------------------------------------------------

n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

#with open("json_arr.gbnf", "r") as file:
with open("json.gbnf", "r") as file:
    grammar_text = file.read()

grammar = LlamaGrammar.from_string(grammar_text)
LLM_MODELS = os.environ['LLM_MODELS']

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path=LLM_MODELS+"zephyr-7b-beta.Q4_K_M.gguf",
    #model_path="/Users/erlebach/data/llm_models/samantha-mistral-instruct-7b.Q4_K_M.gguf",
    #model_path="/Users/erlebach/src/2023/llama-cpp-python/mistral-7b-instruct-v0.1.Q3_K_M.gguf",
    temperature=0.75,
    stop=[],
    max_tokens=2000,
    top_p=1,
    nthreads=16,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

prompt = """
Question: A rap battle between Stephen Colbert and John Oliver
"""
prompt4 = """
System: You are an expert on Shakespeare. 
Question: Generate A python code that counts to 10. 
"""
prompt7 = """
What are the the countries member of NATO and their capitals?. Use json format with keys 'country' and 'capital'.
"""

# Constrain the reply from llm to conform to the grammar
#llm(prompt7)
llm(prompt7, grammar=grammar)

#----------------------------------------------------------------------
