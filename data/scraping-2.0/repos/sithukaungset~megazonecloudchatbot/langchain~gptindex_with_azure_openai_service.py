import os
import openai
from dotenv import load_dotenv
from llama_index import ServiceContext, GPTVectorStoreIndex, LLMPredictor, PromptHelper, SimpleDirectoryReader, load_index_from_storage
from llama_index import StorageContext, load_index_from_storage
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index import LangchainEmbedding

# Load env variables (create .env with OPENAI_API_KEY and OPENAI_API_BASE)
load_dotenv()

# Configure OpenAI API
openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")

deployment_name = "embeddingada002"

# Create LLM via Azure OpenAI Service
llm = AzureOpenAI(deployment_name=deployment_name)
llm_predictor = LLMPredictor(llm=llm)
embedding_llm = LangchainEmbedding(OpenAIEmbeddings())

# Define prompt helper
max_input_size = 3000
num_output = 256
chunk_size_limit = 1000  # token window size per document
max_chunk_overlap = 20  # overlap for each token fragment
prompt_helper = PromptHelper(max_input_size=max_input_size, num_output=num_output,
                             max_chunk_overlap=max_chunk_overlap, chunk_size_limit=chunk_size_limit)

# Read txt files from data directory
documents = SimpleDirectoryReader('data').load_data()
index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor,
                             embed_model=embedding_llm, prompt_helper=prompt_helper)
index.save_to_disk("index.json")

# Query index with a question
response = index.query("What is azure openai service?")
print(response)
