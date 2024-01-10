
# specify the llm model
from langchain.llms import OpenAI
llm=OpenAI(model_name="text-ada-001", temperature=0))

# Alternatively, open-source LLM hosted on Hugging Face
# pip install huggingface_hub
from langchain import HuggingFaceHub
llm = HuggingFaceHub(repo_id = "google/flan-t5-xl")

# HuggingFace Local Pipelines
# pip install transformers
# need to find the appropriate model_id and model is downloaded into ~.cache
from langchain import HuggingFacePipeline
llm = HuggingFacePipeline.from_model_id(model_id="bigscience/bloom-1b7", task="text-generation", model_kwargs={"temperature":0, "max_length":64})

# specify the document loader
from langchain.document_loaders import DirectoryLoader
loader=DirectoryLoader('/Users/band/tmp/workbench/testdir', glob="**/*.txt")

# build an index
from langchain.indexes import VectorstoreIndexCreator
index = VectorstoreIndexCreator().from_loaders([loader])

# get token estimate
# pip install tiktoken
llm.get_num_tokens("prompt? or any? string here")
