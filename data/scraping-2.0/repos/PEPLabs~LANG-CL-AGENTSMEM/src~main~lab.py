import os

from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_core.tools import Tool

"""
This lab will guide you through defining LangChain Agents with ConversationBufferWindowMemory. 
ConversationBufferWindowMemory is a basic way to store conversation history. We can declare this in 
an Agent's "memory" attribute to define a memory mechanism that the LLM can use to complete tasks.

ConversationBufferWindowMemory maintains a record of the conversation's previous interactions, 
limited to the number defined in the "k" attribute. Only the last "k" interactions are stored.
In other words, ConversationBufferWindowMemory maintains a "window" of previous interactions defined by "k". 

https://python.langchain.com/docs/modules/memory/types/buffer_window
"""

"""
Defining our LLM here to include for our chat model, as well as the text input & functions for the tools that our agent
will use. No need to edit these
"""
llm = HuggingFaceEndpoint(
        endpoint_url=os.environ['LLM_ENDPOINT'],
        task="text2text-generation",
        model_kwargs={
            "max_new_tokens": 200
        }
    )
chat_model = ChatHuggingFace(llm=llm)
textInput = """
<|system|>
You are that helpful AI that responds concisely</s>
<|user|>
{userInput}</s>
<|assistant|>
"""

def greeting(input):
    """When the user greets you, greet them back. Nothing else would be said."""
    return chat_model.invoke(textInput.format(userInput=input))


def get_historical_fact(input):
    """If the user asks for a historical fact, give them a concise summary of the topic."""
    return chat_model.invoke(textInput.format(userInput=input))


# TODO: define the second tool that the agent will have access to (get_historical_fact)
tools = [
    Tool.from_function(
        func=greeting,
        name="greeting",
        description="When the user sends a greeting, send a greeting back.",
    ),
    # SECOND TOOL GOES HERE
]


"""
Defining a conversational agent that DOES NOT STORE any conversation history. 
DON'T EDIT THIS CODE, as it's meant to serve as a sample in src/lab.py and proof of concept in src/app.py

First, we define a ConversationBufferWindowMemory object, and store 0 previous interactions in memory. (k=0)
(Spoiler: this is not likely to create an agent that remembers things...)

Next, we use the initialize_agent function, to create an agent providing it with the tools and llm defined above.  

We've defined the agent type as CONVERSATIONAL_REACT_DESCRIPTION. 
This is a more conversational type of agent than the more general-use ZERO-SHOT agents. 
It is designed to hold a conversation while using tools.

By setting verbose=True, we can see what the agent is thinking/doing in the console. 

Finally, we set the memory attribute to the previously defined ConversationBufferWindowMemory object.
"""
memory_no_history = ConversationBufferWindowMemory(memory_key="chat_history", k=0)

agent_executor_no_memory = initialize_agent(
    tools,
    chat_model,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory_no_history,
    handle_parsing_errors=True
)

"""
Defining a conversational agent that STORES 3 previous interactions in memory. 
This is the main task of the lab
"""
# TODO: instantiate a ConversationBufferWindowMemory object that stores 2 previous interactions in memory
memory_with_history = "TODO"

# TODO: define a conversational agent that uses memory_with_history for its memory attribute
agent_executor_with_memory = "TODO"
