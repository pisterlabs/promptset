# refer:https://python.langchain.com/docs/modules/agents/agent_types/openai_multi_functions_agent

#original google AI: https://ai.google.dev/tutorials/python_quickstart?hl=en

import os
from openai import OpenAI
# from langchain.llms import OpenAI
# based on openAI version is higher than 1.0.0[pip install openai==1.6.1]
# python3 localLLM/testOpenAI.py
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenLLM
from langchain.utilities import SerpAPIWrapper
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import DuckDuckGoSearchResults
# Do this so we can see exactly what's going on under the hood
from langchain.globals import set_debug
from langchain.globals import set_verbose
from getpass import getpass


if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key")
else:
    google_api_key = os.environ["GOOGLE_API_KEY"]
    print("GOOGLE_API_KEY already set to:", google_api_key) 


if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Provide your OPENAI_API_KEY")
else:
    openai_api_key = os.environ["OPENAI_API_KEY"]
    print("OPENAI_API_KEY already set to:", openai_api_key)      

    
if "SERPAPI_API_KEY" not in os.environ:
    os.environ["SERPAPI_API_KEY"] = getpass.getpass("Provide your SERPAPI_API_KEY")
else:
    serpapi_api_key = os.environ["SERPAPI_API_KEY"]
    print("SERPAPI_API_KEY already set to:", serpapi_api_key)    


google_api_key = os.environ["GOOGLE_API_KEY"] #
print("GOOGLE_API_KEY=>", google_api_key)

openai_api_key = os.environ["OPENAI_API_KEY"] #
print("OPENAI_API_KEY=>", openai_api_key)

serpapi_api_key = os.environ["SERPAPI_API_KEY"] #
print("SERPAPI_API_KEY=>", serpapi_api_key)

#for debuging refer:https://python.langchain.com/docs/guides/debugging
set_debug(True)
set_verbose(True)
print("==============================================Completed the setup env=========================")

#==============================================Completed the setup env=========================


#==============================================exact logic code=========================
#Initialize Google Generative AI language model
# llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=google_api_key)
# res = llm.invoke( "What are some of the pros and cons of Python as a programming language?")
# print("GoogleGenerativeAI.result=>",res)

# llm = ChatGoogleGenerativeAI(model="gemini-pro")
# result = llm.invoke("Write a ballad about LangChain")
# print("ChatGoogleGenerativeAI.result=>", result)


# Initialize  local OpenAI-LLM language model with LM Studio built server
base_url = os.environ.get("TEST_API_BASE_URL", "http://192.168.0.232:1234/v1")     #This is working as well for LM Studio server
api_key = os.environ.get("OPENAI_API_KEY", "xxxxxxxxx")   # even your local don't use the authorization, but you need to fill something, otherwise will be get exception.
api_key = "xxxx"
openAI_client = OpenAI(
    base_url = base_url,
    # This is the default and can be omitted
    api_key=api_key,

)
#    _strict_response_validation=False,
messages = [{"role": "user", "content": "hi"}]
#model = "/Users/sl6723/.cache/lm-studio/models/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q8_0.gguf"
model = "facebook--opt-1.3b"
chat_completion=openAI_client.chat.completions.create(
    messages=messages, 
    model=model, # this field is currently unused
    stream=False,
)
print(chat_completion.choices[0].message.model_dump())


'''
# ## my own LLM API refer:https://python.langchain.com/docs/integrations/llms/openllm
# # server_url = "http://localhost:3000"  # Replace with remote host if you are running on a remote server
# server_url = "http://192.168.0.232:1234"  # Replace with remote host if you are running on a remote server
# llm = OpenLLM(server_url=server_url)




# Initialize the general search API for search functionality
# Replace <your_api_key> in serpapi_api_key="<your_api_key>" with your actual SerpAPI key.
# search = SerpAPIWrapper()  #SerpAPI need license and charge the fee.
# search = DuckDuckGoSearchResults()  #SerpAPI need license and charge the fee.
search = DuckDuckGoSearchRun()  #SerpAPI need license and charge the fee.       

# Define a list of tools offered by the agent
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful when you need to answer questions about current events. You should ask targeted questions.",
    ),
]
llm=openAI_client
#To make sure that our agent doesn’t get stuck in excessively long loops, we can set max_iterations. We can also set an early stopping method, which will determine our agent’s behavior once the number of max iterations is hit. By default, the early stopping uses method force which just returns that constant string. Alternatively, you could specify method generate which then does one FINAL pass through the LLM to generate an output.
agent_executor = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    max_iterations=20,
    early_stopping_method="generate",
)

#agent_executor.run("What is the weather in LA and SF?(Use search tools)")
#agent_executor.run("What is the weather in NYC today, yesterday, and the day before?")
#agent_executor.run("Could you list Sidan Lu's(Louisiana State University,China,Princeton University,Her current professor is Jason Ren.Her supervisor Yujiao Sun) paper with environmental engineering wastewater treatment?")
#agent_executor.run("what is the lastest progress in membrane distillation technology in 2023?")
#agent_executor.run("What date time is right now with full timezone and area ?")
agent_executor.invoke(
    {
        "input": "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"
    }
)
#==============================================exact logic code=========================

'''