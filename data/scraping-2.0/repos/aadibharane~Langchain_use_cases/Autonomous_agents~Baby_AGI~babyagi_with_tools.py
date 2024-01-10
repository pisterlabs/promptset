'''
BabyAGI with Tools
This notebook builds on top of baby agi, but shows how you can swap out the execution chain. The previous execution chain was just 
an LLM which made stuff up. By swapping it out with an agent that has access to tools, we can hopefully get real reliable information
'''

#Install and Import Required Modules
import os
from collections import deque
from typing import Dict, List, Optional, Any

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from langchain.experimental import BabyAGI

#Connect to the Vector Store
#Depending on what vectorstore you use, this step may look different.
#%pip install faiss-cpu > /dev/null
#%pip install google-search-results > /dev/null
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

import os
import tkinter as tk
from tkinter import messagebox

import openai 
os.environ["OPENAI_API_KEY"] ="OPENAI_API_KEY"
serpapi_key="serpapi_key"

# Define your embedding model
embeddings_model = OpenAIEmbeddings(openai_api_key="OPENAI_API_KEY")
# Initialize the vectorstore as empty
import faiss
# Define your embedding model
def babyagi_tools():
    embeddings_model = OpenAIEmbeddings()
    # Initialize the vectorstore as empty
    import faiss

    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    '''
    Define the Chains
    BabyAGI relies on three LLM chains:
    Task creation chain to select new tasks to add to the list
    Task prioritization chain to re-prioritize tasks
    Execution Chain to execute the tasks
    '''

    from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
    from langchain import OpenAI, SerpAPIWrapper, LLMChain

    todo_prompt = PromptTemplate.from_template(
        "You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}"
    )
    todo_chain = LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)
    search = SerpAPIWrapper(serpapi_api_key="5e4b783d1e905b2992665d83235e27aaa73e103f239fb757b84be1cc2c75c57b")
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events",
        ),
        Tool(
            name="TODO",
            func=todo_chain.run,
            description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!",
        ),
    ]


    prefix = """You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
    suffix = """Question: {task}
    {agent_scratchpad}"""
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["objective", "task", "context", "agent_scratchpad"],
    )

    llm = OpenAI(temperature=0)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True
    )

    #OBJECTIVE = "Write a weather report for Pune today"
    OBJECTIVE = input_entry.get() #"How to become most successful data scientist"

    # Logging of LLMChains
    verbose = False
    # If None, will keep on going forever
    max_iterations: Optional[int] = 3
    baby_agi = BabyAGI.from_llm(
        llm=llm, vectorstore=vectorstore, task_execution_chain=agent_executor, verbose=verbose, max_iterations=max_iterations
    )

    #Run the BabyAGI
    #Now it’s time to create the BabyAGI controller and watch it try to accomplish your objective.

    baby_agi({"objective": OBJECTIVE})

    #print(res)
    # return res

root = tk.Tk()
root.title("Baby AGI with tools")

# Create an entry field for input
input_entry = tk.Entry(root, width=50)
input_entry.pack()

def button_click():
    response = babyagi_tools()
    response_label.config(text=response)

# Create a label to display the response
response_label = tk.Label(root, text="", wraplength=400)
response_label.pack()

# Create a button to trigger the backend function
button = tk.Button(root, text="Run", command=button_click)
button.pack()

# Run the Tkinter event loop
root.mainloop()
