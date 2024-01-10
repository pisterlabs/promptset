# Databricks notebook source
# MAGIC %md
# MAGIC ### LLAMA2 - 13b CPP
# MAGIC
# MAGIC **Author:** Rohan Kataria |
# MAGIC **Email:** imrohankataria@optum.com
# MAGIC
# MAGIC **Cluster Used:** NC6s_v3 ([Azure VM Pricing](https://azureprice.net/vm/Standard_NC6s_v3))
# MAGIC
# MAGIC **References** - 
# MAGIC 1. [djliden - Inference Experiments - LLaMA v2](https://github.com/djliden/inference-experiments/tree/main/llama2)
# MAGIC 2. [abetlen - llama-cpp-python Issue #707](https://github.com/abetlen/llama-cpp-python/issues/707) (Do this step if LLAMA-CPP Doesn't work, install pathspec==0.11.0 via Add Library option using PyPi)
# MAGIC
# MAGIC **Description:**
# MAGIC
# MAGIC This Databricks notebook is an implementation of the LLAMA2 - 13b CPP model, a variant of the LLM (Language Model) architecture. It is designed to run on a Databricks cluster, specifically the NC6s_v3 Azure VM. The code uses the LangChain framework, but you're free to use any other framework that suits your needs.
# MAGIC
# MAGIC The notebook includes instructions for installing necessary libraries and tools, building the LLAMA-CPP-PYTHON with specific arguments, and running the LLM model with specific parameters. It also includes a prompt template and an example of how to run the model with a series of questions.
# MAGIC
# MAGIC Please note that it's important to ensure the model path is correct for your system, and the values for n_gpu_layers, n_batch, and n_ctx are appropriate for your model and GPU VRAM pool.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### Installation

# COMMAND ----------

# MAGIC %pip install langchain transformers torch

# COMMAND ----------

# MAGIC %md
# MAGIC Important: Install nvidia cuda toolkit --- needed driver to build LLAMA-CPP

# COMMAND ----------

!apt-get install nvidia-cuda-toolkit -y

# COMMAND ----------

# MAGIC %md
# MAGIC Important: build LLAMA-CPP-PYTHON with "-DLLAMA_CUBLAS=on" FORCE_CMAKE=1

# COMMAND ----------

!CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade llama-cpp-python

# COMMAND ----------

from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Downloading Model

# COMMAND ----------

!pwd
!ls

# COMMAND ----------

!wget https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q5_K_M.gguf

# COMMAND ----------

# MAGIC %md
# MAGIC #### Working

# COMMAND ----------

# MAGIC %md
# MAGIC ##### PROMPT

# COMMAND ----------

template = """Question: {question}

Answer: """

prompt = PromptTemplate(template=template, input_variables=["question"])

# COMMAND ----------

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# COMMAND ----------

# MAGIC %md
# MAGIC ##### LLM
# MAGIC - https://api.python.langchain.com/en/latest/llms/langchain.llms.llamacpp.LlamaCpp.html?highlight=llamacpp#langchain.llms.llamacpp.LlamaCpp

# COMMAND ----------

n_gpu_layers = 120  # Change this value based on your model and your GPU VRAM pool.
n_batch = 1024  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU. 
n_ctx = 4096 # context length

# n_gpu_layers=1 means only one layer of the model will be loaded into GPU memory.
# n_batch=512 sets the number of tokens the model should process in parallel.
# n_ctx=2048 sets the token context window, meaning the model will consider a window of 2048 tokens at a time.
# f16_kv=True means the model will use half-precision for the key/value cache, which can be more memory efficient.

# Make sure the model path is correct for your system!
llm_gpu = LlamaCpp(
    model_path='llama-2-13b-chat.Q5_K_M.gguf',
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=n_ctx,
    f16_kv=True,
    # temperature=0.9,
    # top_k=40,
    # top_p=1,
    callback_manager=callback_manager,
    verbose=True, # Verbose is required to pass to the callback manager
)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### CHAIN

# COMMAND ----------

llm_chain = LLMChain(prompt=prompt, llm=llm_gpu)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### QnA:

# COMMAND ----------

question = "What is your name?"
llm_chain.run(question)

# COMMAND ----------

question = "What is LLAMA?"
llm_chain.run(question)

# COMMAND ----------

question = "Write Python function to add 2 numbers"
llm_chain.run(question)

# COMMAND ----------

question = "What is winterfell in game of thrones"
llm_chain.run(question)

# COMMAND ----------

question = "How to tell I have covid?"
llm_chain.run(question)

# COMMAND ----------

question = "Suggest me good habits to focus on 8hours of sleep"
llm_chain.run(question)
