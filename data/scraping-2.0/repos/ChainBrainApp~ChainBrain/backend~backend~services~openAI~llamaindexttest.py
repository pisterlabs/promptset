from langchain import OpenAI
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader
# from helpers.schemas import NFT_Marketplace
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.
# documents = SimpleDirectoryReader('docs').load_data()
# index = GPTSimpleVectorIndex(documents)

# # response = index.query("""You are an AI that helps write GraphQL queries on the Graph Protocol. In the coming prompts I'll feed you questions that you need to turn into graphQL queries that work. Show only code and do not use sentences. Note that it's important that if you don't have some specific data (like dates or IDs), just add placeholders. Show only code and do not use sentences. 
# # What is the past 24 hour volume of NFTs?""")
# # What is the last 7 days of trading pair volume on Uniswap sorted by USD volume?
# response = index.query("""You are an AI that helps write GraphQL queries on the Graph Protocol. In the coming prompts I'll feed you questions that you need to turn into graphQL queries that work. Show only code and do not use sentences. Note that it's important that if you don't have some specific data (like dates or IDs), just add placeholders. Show only code and do not use sentences. 
# What is the proposal with the most votes?""")
# print(response)

from llama_index import LLMPredictor, GPTSimpleVectorIndex, PromptHelper
documents = SimpleDirectoryReader('docs/lending').load_data()
index = GPTSimpleVectorIndex(documents)
# define LLM
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

# define prompt helper
# set maximum input size
max_input_size = 4096
# set number of output tokens
num_output = 256
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

index = GPTSimpleVectorIndex(
    documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
)

# response = index.query("""You are an AI that helps write GraphQL queries on the Graph Protocol. In the coming prompts I'll feed you questions that you need to turn into graphQL queries that work. Show only code and do not use sentences. Note that it's important that if you don't have some specific data (like dates or IDs), just add placeholders. Show only code and do not use sentences. 
# What is the total trading volume by tier?""")
response = index.query("""You are an AI that helps write GraphQL queries on the Graph Protocol. In the coming prompts I'll feed you questions that you need to turn into graphQL queries that work. Show only code and do not use sentences. Note that it's important that if you don't have some specific data (like dates or IDs), just add placeholders. Show only code and do not use sentences. 
What is the total amount borrowed?""")
print(response)