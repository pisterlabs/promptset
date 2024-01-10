import os
os.environ['CUBLAS_LOGINFO_DBG'] = '1'


from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
import httpx
from llama_cpp.llama import Llama, LlamaGrammar

#template = """Question: {question}

#Answer: Let's work this out in a step by step way to be sure we have the right answer."""

#prompt = PromptTemplate(template=template, input_variables=["question"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# ----------------------------------------------------------------------

n_gpu_layers = 20  # Change this value based on your model and your GPU VRAM pool.
n_batch = 128  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

with open("json_arr.gbnf", "r") as file:
#with open("json.gbnf", "r") as file:
    grammar_text = file.read()

grammar = LlamaGrammar.from_string(grammar_text)
LLM_MODELS = os.environ['LLM_MODELS']

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path=LLM_MODELS+"zephyr-7b-beta.Q4_K_M.gguf",
    #model_path="/Users/erlebach/data/llm_models/samantha-mistral-instruct-7b.Q4_K_M.gguf",
    ##model_path="/Users/erlebach/data/em_german_leo_mistral.Q3_K_M.gguf",
    n_gpu_layers=n_gpu_layers,
    # max_tokens=24,  # works
    n_ctx=2048,
    stop=[],
    max_tokens=1000,
    n_threads=12,
    temperature=0.4,  # also works with 0.0 (0.01 is safer)
    f16_kv=True,
    n_batch=n_batch,
    callback_manager=callback_manager,
    verbose=False,  
)

prompts = []
prompt1 = """ Question: Assume base 18 defined by (0,1,2,...,9,A,B,C,D,E,F,G,H). What is the three numbers that logically follow 17 (which is a number expressed in base 18). Express the final answer in base 18. Proceed step by step. """
prompt3 = """ Question: Assume all numbers in base 18 defined by (0,1,2,...,9,A,B,C,D,E,F,G,H). What is the 1G + A3? """

prompt2 = """
Question: Given the four concepts "even numbers, odd numbers, divisible by 7, prime numbers", to what concept would you ascribe to the number 16? Reason step by step. 
"""

prompt4 = """
Question: A rap battle between Stephen Colbert and John Oliver
"""

prompt5 = """
You are an expert Python programmer. 
Generate a python code that generates the first 10 prime numbers. 
"""

prompt6 = """
What are the capitals of all countries member of NATO, expressed in json format, with keys "country" and "capital".
"""
prompt6 = """
What are the all the countries member of NATO, expressed as a json list? 
"""

prompt7 = """
What are the the countries member of NATO and their capitals?. Use json format with keys 'country' and 'capital'. 
"""
prompt8 = """
What are the countries member of NATO, with keys 'pais' (french) and 'capitale' (french)?"
"""

#prompt6 = """
#List the capitals of the countries member of NATO, expressed as a json list. 
#"""

#prompt6 = """
#List the capitals of the countries member of NATO?
#"""
# What are the capitals of all countries member of NATO?

# ----------------------------------------------------------------------

# How does batch work?
#llm(prompt7)

# Constrain the reply from llm to conform to the grammar
prompt8 = messages = [
    {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate"},
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"}
]

prompt = ""
for message in messages:
    prompt += f"{message['role']}: {message['content']}\n"

llm(prompt, grammar=grammar)
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
