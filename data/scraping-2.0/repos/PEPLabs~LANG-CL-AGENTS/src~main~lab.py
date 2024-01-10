import os

from langchain.agents import initialize_agent, AgentType
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_core.tools import Tool

"""
All requests to the LLM require some form of a key.
Other sensitive data has also been hidden through environment variables.
"""

"""
We will create an agent that can deduce the length of a word or the cube of a number.
The agent will decide on one process or another by matching a given task with the description of its tools.

https://python.langchain.com/docs/modules/agents/
"""

llm = HuggingFaceEndpoint(
    endpoint_url=os.environ['LLM_ENDPOINT'],
    task="text2text-generation",
    model_kwargs={
        "max_new_tokens": 200
    }
)

"""
The two functions below will serve as tools for the agent. 
In other words, they are processes that the agent can choose from to complete a given task.
However, the agent won't know which tool to use if they aren't described well. 
"""
def get_word_length(word) -> int:
    # TODO: write a description of what this tool does
    """TODO"""
    return len(word)


def get_cube_of_number(number) -> int:
    # TODO: write a description of what this tool does
    """TODO"""
    return pow(int(number), 3)

"""
Here, we are defining the tools that the agent will have access to. 
"""
# TODO: finish defining the second tool that the agent will have access to (get_word_length)
tools = [
    Tool.from_function(
        func=get_cube_of_number,
        name="get_cube_of_number",
        description="finds the cube of a number",
    ),
    # TODO
]

"""
We can now create the agent! No need to edit the code below.
We are using the initialize_agent function, providing it with the tools and llm defined above.  

We've defined the agent type as ZERO_SHOT_REACT_DESCRIPTION. 
This is the most general type of agent. It determines which tool to use solely based on the tool descriptions.

By setting verbose=True, we can see what the agent is thinking/doing in the console. 

Finally, by setting return_intermediate_steps=True, we can access the intermediate steps of the agent.
This is useful for debugging and writing tests.
"""
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    return_intermediate_steps=True,
)
