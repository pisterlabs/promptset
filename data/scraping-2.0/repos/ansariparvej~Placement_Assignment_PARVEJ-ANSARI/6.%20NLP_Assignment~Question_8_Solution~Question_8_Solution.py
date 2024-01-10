# !pip install llama_index langchain

# File Path (file name => ./data/llama.rtf)
file_path = './data'

# Import Required libraries
from llama_index import LLMPredictor, GPTVectorStoreIndex, PromptHelper, ServiceContext
from llama_index import SimpleDirectoryReader
from langchain import OpenAI
import os

# Authentication key
os.environ['OPENAI_API_KEY'] = "<YOUR OPEN-AI API-KEY>"

# Load you data into 'Documents' a custom type by LlamaIndex
documents = SimpleDirectoryReader(file_path).load_data()


# Setup Custom LLM

# Define LLM
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.1, model_name="text-davinci-002"))

# Define prompt helper

# set maximum input size
max_input_size = 4096
# set number of output tokens
num_output = 256
# set maximum chunk overlap
max_chunk_overlap = 20

prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

# Query your index!
query_engine = index.as_query_engine()
response = query_engine.query("What do you think of Facebook's LLaMa?")
print(response)

# Response ==>  I think it's a great idea!
