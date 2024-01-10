# refer:https://python.langchain.com/docs/modules/agents/agent_types/self_ask_with_search

import os
from langchain.agents import AgentType, Tool, initialize_agent
from langchain_google_genai import GoogleGenerativeAI
from langchain.llms import OpenAI
from langchain.llms import OpenLLM
from langchain.utilities import SerpAPIWrapper
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import DuckDuckGoSearchResults
# Do this so we can see exactly what's going on under the hood
from langchain.globals import set_debug
from getpass import getpass


set_debug(True)

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key")
else:
    google_api_key = os.environ["GOOGLE_API_KEY"]
    print("GOOGLE_API_KEY already set to:", google_api_key) 

#==============================================Completed the setup env=========================


#==============================================exact logic code=========================
#Initialize language model
# llm = OpenAI(temperature=0)
#Initialize Google Generative AI language model
# llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=google_api_key)

## my own LLM API refer:https://python.langchain.com/docs/integrations/llms/openllm
server_url = "http://localhost:3000"  # Replace with remote host if you are running on a remote server
# server_url = "http://192.168.0.232:1234"  # Replace with remote host if you are running on a remote server
llm = OpenLLM(server_url=server_url)
# llm = OpenLLM(
#     model_name="dolly-v2",
#     model_id="databricks/dolly-v2-3b",
#     temperature=0.94,
#     repetition_penalty=1.2,
# )
    
# Initialize the general search API for search functionality
# Replace <your_api_key> in serpapi_api_key="<your_api_key>" with your actual SerpAPI key.
# search = SerpAPIWrapper()  #SerpAPI need license and charge the fee.
# search = DuckDuckGoSearchResults()  #SerpAPI need license and charge the fee.
search = DuckDuckGoSearchRun()  #SerpAPI need license and charge the fee.       

# Define a list of tools offered by the agent
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search",
    )
]


self_ask_with_search = initialize_agent(
    tools,
    llm,
    agent=AgentType.SELF_ASK_WITH_SEARCH,#SELF_ASK_WITH_SEARCH,   #OPENAI_FUNCTIONS
    verbose=True,
)

# self_ask_with_search.invoke(
#     {
#         "input": "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"
#     }
# )

self_ask_with_search.run(
    "What is the hometown of the reigning men's U.S. Open champion?"
)

#==============================================exact logic code=========================