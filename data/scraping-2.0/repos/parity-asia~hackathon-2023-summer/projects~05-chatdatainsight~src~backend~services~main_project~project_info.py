
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from services.main_project.moonbeam.moonbeam import moonbeam_query_engine
from core.config import Config

MODEL_NAME = Config.MODEL_NAME
LLM = OpenAI(model_name=MODEL_NAME, temperature=0)  

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI,  LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re

# Set up the base template
PROJECT_AGENT_PROMPT_V1 = """Assume you're an investment research analyst with in-depth expertise in the areas of blockchain, web3, and artificial intelligence. 
Now, your task is to use this knowledge to answer the upcoming questions in the most professional way. if you find one or more image links, the answer must contains them,they are come from data
Before responding, carefully consider the purpose of each tool and whether it matches the question you need to answer. 
It is advisable to first take a holistic look at the names of the tools, and then make a selection.
You can use the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: you should input {input} as param of tool
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times),if you find one or more image links, the answer must contains them,they are come from data
Thought: I now know the final answer
Final Answer: the final answer to the original input question,if you find one or more image links, the answer must contains them

Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s

Question: {input}
{agent_scratchpad}"""

# Set up the base template
PROJECT_AGENT_PROMPT_V2 = """Assume you're an investment research analyst with in-depth expertise in the areas of blockchain, web3, and artificial intelligence.
Now, your task is to use this knowledge to answer the upcoming questions in the most professional way. Before responding, carefully consider the purpose of each tool and whether it matches the question you need to answer.
You can use the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Action: the action to take, should be one of [{tool_names}]
Action Input: you should input {input} as the parameter of the selected tool

Begin! Choose the appropriate tool to obtain the answer and provide the observed result as your final response.

Question: {input}
{agent_scratchpad}"""

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)

        
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

def project_agent(question: str):

    

    llm = LLM 

    # Define which tools the agent can use to answer user queries
    tools = [
        Tool(
            name="Project Ecosystem Development",
            func=moonbeam_query_engine,
            description="Useful for answering questions about the development of a project's ecosystem, such as 'How many DApps are there on Moonbeam?' or 'How many stablecoins are available on Moonbeam?' or 'What are the recent major technical updates on Moonbeam?'"
        ),
    ]

    prompt = CustomPromptTemplate(
        template=PROJECT_AGENT_PROMPT_V2,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps"]
    ) 

    output_parser = CustomOutputParser()

    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names
    )

    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    result = agent_executor.run(question)

    print("INFO:     Project Agent Result:", result)

    return result