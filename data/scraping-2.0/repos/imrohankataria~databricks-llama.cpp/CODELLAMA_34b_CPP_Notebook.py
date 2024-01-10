# Databricks notebook source
# MAGIC %md ### CODELLAMA - 34b CPP
# MAGIC
# MAGIC **Author:** Rohan Kataria |
# MAGIC **Email:** imrohankataria@optum.com
# MAGIC
# MAGIC **Cluster Used:** NC12s_v3 ([Azure VM Pricing](https://azureprice.net/vm/Standard_NC12s_v3))
# MAGIC
# MAGIC **Models Used:** 
# MAGIC 1. [Phind-CodeLlama-34B-v2-GGUF](https://huggingface.co/TheBloke/Phind-CodeLlama-34B-v2-GGUF)
# MAGIC 2. [CodeLlama-34B-GGUF](https://huggingface.co/TheBloke/CodeLlama-34B-GGUF)
# MAGIC
# MAGIC **References** - 
# MAGIC 1. [djliden - Inference Experiments - LLaMA v2](https://github.com/djliden/inference-experiments/tree/main/llama2)
# MAGIC 2. [abetlen - llama-cpp-python Issue #707](https://github.com/abetlen/llama-cpp-python/issues/707) (Do this step if LLAMA-CPP Doesn't work, install pathspec==0.11.0 via Add Library option using PyPi)
# MAGIC 3. [Langchain Code Understanding](https://python.langchain.com/docs/use_cases/code_understanding)
# MAGIC
# MAGIC **Description:**
# MAGIC
# MAGIC This Databricks notebook is an implementation of the CODELLAMA - 34B CPP model, a variant of the LLM (Language Model) architecture. It is designed to run on a Databricks cluster, specifically the NC12s_v3 Azure VM. The code uses the LangChain framework, but you're free to use any other framework that suits your needs.
# MAGIC
# MAGIC The notebook includes instructions for installing necessary libraries and tools, building the LLAMA-CPP-PYTHON with specific arguments, and running the LLM model with specific parameters. It also includes a prompt template and an example of how to run the model with a series of questions.
# MAGIC
# MAGIC Please note that it's important to ensure the model path is correct for your system, and the values for n_gpu_layers, n_batch, and n_ctx are appropriate for your model and GPU VRAM pool.
# MAGIC

# COMMAND ----------

# MAGIC %md #### Installation

# COMMAND ----------

# MAGIC %pip install langchain transformers torch

# COMMAND ----------

# MAGIC %md Important: Install nvidia cuda toolkit --- needed driver to build LLAMA-CPP

# COMMAND ----------

!apt-get install nvidia-cuda-toolkit -y

# COMMAND ----------

# MAGIC %md Important: build LLAMA-CPP-PYTHON with "-DLLAMA_CUBLAS=on" FORCE_CMAKE=1

# COMMAND ----------

!CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade llama-cpp-python

# COMMAND ----------

from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain 
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.question_answering import load_qa_chain

# COMMAND ----------

# MAGIC %md ##### Downloading Model

# COMMAND ----------

!pwd
!ls

# COMMAND ----------

# !wget https://huggingface.co/TheBloke/CodeLlama-34B-GGUF/resolve/main/codellama-34b.Q5_K_M.gguf

# COMMAND ----------

!wget https://huggingface.co/TheBloke/Phind-CodeLlama-34B-v2-GGUF/resolve/main/phind-codellama-34b-v2.Q5_K_M.gguf

# COMMAND ----------

# !ls
# !rm phind-codellama-34b-v2.Q5_K_M.gguf.1 phind-codellama-34b-v2.Q5_K_M.gguf

# COMMAND ----------

# MAGIC %md #### Working

# COMMAND ----------

# MAGIC %md ##### PROMPT

# COMMAND ----------

template = """Give only code output to keep answers short

Question: {question}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["question"])

# COMMAND ----------

# MAGIC %md ##### LLM
# MAGIC - https://api.python.langchain.com/en/latest/llms/langchain.llms.llamacpp.LlamaCpp.html?highlight=llamacpp#langchain.llms.llamacpp.LlamaCpp

# COMMAND ----------

n_gpu_layers = 120  # Change this value based on your model and your GPU VRAM pool.
n_batch = 1024  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU. 
n_ctx = 5000 # context length

# n_gpu_layers=1 means only one layer of the model will be loaded into GPU memory.
# n_batch=512 sets the number of tokens the model should process in parallel.
# n_ctx=2048 sets the token context window, meaning the model will consider a window of 2048 tokens at a time.
# f16_kv=True means the model will use half-precision for the key/value cache, which can be more memory efficient.

# Make sure the model path is correct for your system!
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    # model_path="codellama-34b.Q5_K_M.gguf",
     model_path="phind-codellama-34b-v2.Q5_K_M.gguf",
    n_ctx=n_ctx,
    n_gpu_layers=n_batch,
    n_batch=n_batch,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,
)

# COMMAND ----------

# MAGIC %md ##### CHAIN

# COMMAND ----------

llm_chain = LLMChain(prompt=prompt, llm=llm)

# COMMAND ----------

# MAGIC %md ##### QnA:

# COMMAND ----------

question = "Write a Code to add 2 numbers in python"
llm_chain.run(question)

# COMMAND ----------

#CODELLAMA Without Fine-tune gives an answer with YAML (More correct)
question = "Write Strimzi Kafka Connect JDBC Source Config"
llm_chain.run(question)

# COMMAND ----------

#This is a leetcode question
question = """Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.
The overall run time complexity should be O(log (m+n))."""
llm_chain.run(question)

# COMMAND ----------

question = """Create Snowflake Stored procedure to flatten json data from snowflake kafka connect"""
llm_chain.run(question)
