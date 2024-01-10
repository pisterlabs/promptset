#CSV Agent
'''
This notebook shows how to use agents to interact with a csv. It is mostly optimized for question answering.

NOTE: this agent calls the Pandas DataFrame agent under the hood, which in turn calls the Python agent, which executes 
LLM generated Python code - this can be bad if the LLM generated Python code is harmful. Use cautiously.
'''
import os

os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
def csv_agent():
    agent = create_csv_agent(OpenAI(temperature=0), 'titanic.csv', verbose=True)
    agent.run("how many rows are there?")

    agent.run("how many people have more than 3 siblings")
    agent.run("whats the square root of the average age?")

csv_agent()
#Multi CSV Example
#This next part shows how the agent can interact with multiple csv files passed in as a list.
# agent = create_csv_agent(OpenAI(temperature=0), ['titanic.csv', 'titanic_age_fillna.csv'], verbose=True)
# agent.run("how many rows in the age column are different?")



# import os
# from langchain.agents import create_csv_agent
# from langchain.llms import OpenAI

# os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

# # Single CSV file example
# agent = create_csv_agent(OpenAI(temperature=0), 'titanic.csv', verbose=True)
# agent.run("how many rows are there?")
# agent.run("how many people have more than 3 siblings?")
# agent.run("what's the square root of the average age?")

# # Multi CSV example
# csv_files = ['titanic.csv', 'titanic_age_fillna.csv']
# agents = []
# for file in csv_files:
#     agent = create_csv_agent(OpenAI(temperature=0), file, verbose=True)
#     agents.append(agent)

# for agent in agents:
#     agent.run("how many rows in the age column are different?")
