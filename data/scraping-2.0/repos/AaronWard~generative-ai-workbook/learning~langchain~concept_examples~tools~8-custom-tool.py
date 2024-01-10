"""
When constructing your own agent, you will need to provide it with a list of Tools that it can use. 
Besides the actual function that is called, the Tool consists of several components:
    - name (str), is required and must be unique within a set of tools provided to an agent
    - description (str), is optional but recommended, as it is used by an agent to determine tool use
    - return_direct (bool), defaults to False
    - args_schema (Pydantic BaseModel), is optional but recommended, can be used to provide more information or validation for expected parameters.


https://python.langchain.com/en/latest/modules/agents/tools/custom_tools.html


"""

import os 
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from langchain import OpenAI 
from langchain.tools import BaseTool
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

load_dotenv(Path('./.env'))
llm = OpenAI(temperature=0)


# Create custom tool:
def game_of_thrones_quote(input=""):
    """ This tool says a GOT quote"""
    return 'Winter is coming'

'''
NOTE: If the LLM isn't using your tools, you will need to play around with 
the wording to make it take your tool into consideration.

This depends on yout description for the tool AND telling it how to act
in the prompt.
'''
custom_tool = Tool(
    name="GameOfThronesTool",
    func=game_of_thrones_quote, 
    description="This tool is used to give a game of thrones quote"
)
    
tools = [custom_tool]


# conversational agent memory
memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=3,
    return_messages=True
)

# create our agent
conversational_agent = initialize_agent(
    agent='conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=memory
)

# PROMPT ENGINEERING: 
# following is to be used with chat-conversational-react-description
# however its broken right now
# conversational_agent.agent.llm_chain.prompt.messages[0].prompt.template = "" new_prompt_template

conversational_agent.agent.llm_chain.prompt.template = conversational_agent.agent.llm_chain.prompt.template + "\n \nGive the shortest answer possible."
# print(conversational_agent.agent.llm_chain.prompt.template)

# Ask the model
conversational_agent("What is a Game Of Thrones Quote?")


'''
Assistant is a large language model trained by OpenAI.
Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

TOOLS:
------

Assistant has access to the following tools:

> GameOfThronesTool: This tool is used to give a game of thrones quote

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [GameOfThronesTool]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
AI: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
 
Give the shortest answer possible.


> Entering new AgentExecutor chain...

Thought: Do I need to use a tool? Yes
Action: GameOfThronesTool
Action Input: Quote
Observation: Winter is coming
Thought:

AI: Winter is coming.

> Finished chain.
'''