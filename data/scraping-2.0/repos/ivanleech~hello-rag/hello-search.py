import os
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from dotenv import load_dotenv
from langchain.agents import initialize_agent
from langchain.llms import CTransformers

load_dotenv()

# initialize the search chain
search = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))

llm = CTransformers(
    model="/home/ivanleech/apps/github_new/llm/zephyr-7b-beta.Q4_K_M.gguf",
    model_type="mistral",
    lib="avx2",
)

# create a search tool
tools = [Tool(name="Intermediate Answer", func=search.run, description="google search")]

# initialize the search enabled agent
self_ask_with_search = initialize_agent(
    tools, llm, agent="self-ask-with-search", verbose=True, max_iterations=3, early_stopping_method="generate",
)

self_ask_with_search("Who is Ng Kok Song?")
