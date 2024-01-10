# Databricks notebook source
# MAGIC %pip install langchain

# COMMAND ----------

# MAGIC %pip install "accelerate>=0.16.0,<1" "transformers[torch]>=4.28.1,<5" "torch>=1.13.1,<2"

# COMMAND ----------

# MAGIC %pip install git+https://github.com/huggingface/transformers

# COMMAND ----------

import torch
from transformers import pipeline
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline

# COMMAND ----------

import torch
from transformers import pipeline

device = 0 if torch.cuda.is_available() else -1
generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16, trust_remote_code=True, device=device)# device_map="auto"

# COMMAND ----------


# template for an instrution with no input
prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}")

# template for an instruction with input
prompt_with_context = PromptTemplate(
    input_variables=["instruction", "context"],
    template="{instruction}\n\nInput:\n{context}")

hf_pipeline = HuggingFacePipeline(pipeline=generate_text)

llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)
llm_context_chain = LLMChain(llm=hf_pipeline, prompt=prompt_with_context)


# COMMAND ----------

llm_chain.predict(instruction="Is Arcosa Inc greenwashing ? ")

# COMMAND ----------

llm_chain.predict(instruction="Is Adient company greenwashing ? ")

# COMMAND ----------

llm_chain.predict(instruction="Is Archer Daniels Midland company greenwashing ? ")

# COMMAND ----------

llm_chain.predict(instruction="Is Delta Airlines company greenwashing ? ")

# COMMAND ----------

llm_chain.predict(instruction="Is Essential Utilities company greenwashing ? ")

# COMMAND ----------

llm_chain.predict(instruction="Is NiSource company greenwashing ? ")

# COMMAND ----------

# llm_chain.predict(instruction="Is Arcosa Inc greenwashing ? ")

# COMMAND ----------

# context = """George Washington (February 22, 1732[b] – December 14, 1799) was an American military officer, statesman,
# and Founding Father who served as the first president of the United States from 1789 to 1797."""

# print(llm_context_chain.predict(instruction="When was George Washington president?", context=context).lstrip())

# COMMAND ----------

# PROMPT ENGINEERING 
# llm_context_chain.predict(instruction="When was George Washington president? ", context=context).lstrip()

# COMMAND ----------

# df = spark.read.table('gtp_csv').toPandas()

# COMMAND ----------

# torch.cuda.empty_cache()
# import gc
# del variables
# gc.collect()

# COMMAND ----------

# llm_context_chain.predict(instruction="Is this company greenwashing? ", context=df.iloc[2].transcript)
# OutOfMemoryError: CUDA out of memory. Tried to allocate 26.20 GiB (GPU 0; 15.75 GiB total capacity; 5.95 GiB already allocated; 8.73 GiB free; 6.08 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF


# COMMAND ----------



# COMMAND ----------
