import os
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI

from dotenv import load_dotenv
load_dotenv()


api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(openai_api_key=api_key, temperature=0)


tools = load_tools(['google-serper'])#, llm=llm) #, 'wikipedia', 'arxiv'
print(tools[0].name, tools[0].description)

# Create the agent
agent = initialize_agent(
    tools,
    llm, 
    agent='zero-shot-react-description',
    verbose=True)


# keword explenation
# item = agent.run("explain 'Multiple Node-centered Subgraphs' in 5 sentences.Give the url link to the source  as [more info](url link). and 2 usecase example in")

item = agent.run('give the link to pdf of paper: Chen Zhao. 2021. Fairness-Aware Multi-Task and Meta Learning . Ph. D. Disserta-\ntion.')

print(item)


