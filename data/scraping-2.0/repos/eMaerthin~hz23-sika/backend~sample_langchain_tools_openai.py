from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.agents import OpenAIFunctionsAgent
from langchain.agents import load_tools
from langchain.retrievers import ChatGPTPluginRetriever
from langchain.agents import AgentExecutor
from langchain.tools import AIPluginTool
from langchain.requests import RequestsWrapper
from langchain.agents import create_openapi_agent
from langchain.llms.openai import OpenAI
from langchain.agents.agent_toolkits import OpenAPIToolkit
from langchain.tools.json.tool import JsonSpec
import requests
from langchain.tools import StructuredTool
from langchain.memory import ConversationBufferMemory

from langchain.prompts import MessagesPlaceholder

from langchain.agents import AgentType, initialize_agent

from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.tools.playwright.utils import (
    create_async_playwright_browser,
    create_sync_playwright_browser, # A synchronous browser is available, though it isn't compatible with jupyter.
)
from langchain.agents import initialize_agent, Tool


from dotenv import load_dotenv
import os, yaml
load_dotenv()  # take environment variables from .env.
RETRIEVER = ChatGPTPluginRetriever(url="http://0.0.0.0:8080", bearer_token=os.getenv("BEARER_TOKEN"))

def find_knowledge(input:str) -> str:
    """
    Returns top_k=10 closest matches from the sika database given the user_prompt.
    :param input: user question. The user might be a customer (construction engineer) or sales person
    :return: content of 10 closes matches from the Sika Knowledge database to the given {input}.
    """
    ret_val = ""
    for i, elem in enumerate(RETRIEVER.get_relevant_documents(input, top_k=10), 1):
        ret_val += f"##Match{i}: {elem.page_content}##\n"
    print(ret_val)
    return ret_val

tools = [
    Tool(
        name="Intermediate Answer",
        func=find_knowledge,
        description="useful for when you need to ask with search",
    )
]
llm = ChatOpenAI(temperature=0)

self_ask_with_search = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
self_ask_with_search.run(
    "How to seal leakages on cracked concrete?"
)

"""
tool = StructuredTool.from_function(find_knowledge)#, name="find_knowledge", description=find_knowledge.__doc__)
llm = ChatOpenAI(temperature=0)
tools = [tool]
chat_history = MessagesPlaceholder(variable_name="chat_history")

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

sync_browser = create_sync_playwright_browser()
browser_toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
tools = browser_toolkit.get_tools()


agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    agent_kwargs = {
        "memory_prompts": [chat_history],
        "input_variables": ["input", "agent_scratchpad", "chat_history"]
    }
)
response = agent_chain.run(input="How to seal leakages on cracked concrete?")
print(response)
"""
"""
agent = initialize_agent(
    tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

print(agent.run("How to seal leakages on cracked concrete?", verbose=True))
"""
"""
chat = ChatOpenAI(
    model_name=os.getenv("OPENAI_CHAT_MODEL_NAME"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
requests_wrapper = RequestsWrapper(headers={"Authorization": f"Bearer {os.getenv('BEARER_TOKEN')}"})

llm = ChatOpenAI(temperature=0)

p = 'http://localhost:8080/openapi.json'
response = requests.get(p)
data = yaml.load(response.text, Loader=yaml.FullLoader)
json_spec = JsonSpec(dict_=data, max_value_length=4000)
openapi_toolkit = OpenAPIToolkit.from_llm(llm, json_spec, requests_wrapper, verbose=True)

agent = create_openapi_agent(llm=llm, toolkit=openapi_toolkit, verbose=True)

output = agent.run("What is cementitious waterproofing mortar?")

print(output)
"""


"""
#retriever = ChatGPTPluginRetriever(url="http://0.0.0.0:8080", bearer_token=os.getenv("BEARER_TOKEN"))

# Tool
tool = AIPluginTool.from_plugin_url("http://0.0.0.0:8080/.well-known/ai-plugin.json")#, bearer_token=os.getenv("BEARER_TOKEN"))
llm = ChatOpenAI(temperature=0)
tools = load_tools(["requests_all"])
tools += [tool]

agent_chain = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
agent_chain.run("what t shirts are available in klarna?")



tools = load_tools(["requests_all"])
tools += [tool]


system_message = SystemMessage(content="You are a search assistant.")
prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
search_agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=search_agent, tools=tools, verbose=True, requests_wrapper=requests_wrapper
)
output = agent_executor.run("What is cementitious waterproofing mortar?")

print(output)
"""