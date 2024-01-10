from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from serpapi import GoogleSearch
import openai
from os.path import join, dirname
from keybert import KeyBERT
import os
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), "process.env")
load_dotenv(dotenv_path)

openai=os.getenv("OPENAI_API_KEY")
serpapi=os.getenv("SERPAPI_API_KEY")

os.environ['OPENAI_API_KEY']=str(openai)
os.environ['SERPAPI_API_KEY'] =str(serpapi)


valid_keywords = ["agriculture", "farming", "crop", "farmer", "livestock", "irrigation", "soil", "harvest",
    "cultivation", "agronomy", "sustainable farming", "organic farming", "crop rotation",
    "pest control", "fertilizers", "agribusiness", "rural development", "agricultural technology",
    "precision farming", "agroforestry", "horticulture", "animal husbandry", "aquaculture",
    "agricultural machinery", "food security", "biodiversity", "crop yield",
    "genetic engineering in agriculture", "greenhouse farming", "agricultural economics",
    "rural livelihoods", "agrochemicals", "agroprocessing", "farm management",
    "agricultural research", "sustainable agriculture practices", "food production",
    "land conservation", "livestock management", "agricultural policy", "agrotourism",
    "wheat", "rice", "maize", "barley", "oats", "soybeans", "potatoes", "tomatoes",
    "carrots", "lettuce", "apples", "bananas", "grapes", "oranges", "strawberries", "diseases"]


invalid_keywords = [
    "porn", "xxx", "adult", "explicit", "gambling", "casino", "drugs", "illegal",
    "hate", "violence", "weapons", "offensive", "inappropriate", "spam", "scam",
    "fraud", "malware", "virus", "hack", "cheat", "unrelated", "irrelevant", "fake",
    "hoax", "misleading", "deceptive","sex","cricket","football"
]

map_valid_keywords=map(lambda x:x,valid_keywords)
map_invalid_keywords=map(lambda x:x,invalid_keywords)

search = SerpAPIWrapper()
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    )
]


prefix = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:"""
suffix = """Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Args"

Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools, 
    prefix=prefix, 
    suffix=suffix, 
    input_variables=["input", "agent_scratchpad"]
)


def check_validity(message):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(message)
    result_keywords = [item[0] for item in keywords]
    # print(result_keywords)
    valid_count=0
    invalid_count=0
    for keys in result_keywords:
        valid_present = any(keys in value for value in list(valid_keywords))
        invalid_present=any(keys in value for value in list(invalid_keywords))
        
        if(valid_present):
            valid_count+=1
        if(invalid_present):
            invalid_count+=1
    
    if(invalid_count):
        return False
    if(valid_count>=0):
        return True





def create_chatbot_executor(message):
    if(check_validity(message)):
        # print("Valid question")
        llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
        tool_names = [tool.name for tool in tools]
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
        result=agent_executor.run(message)
        return result
    else:
        print("This is the invalid question")

