import time
from typing import Callable
from typing import Union
from langchain import LLMChain, OpenAI, SerpAPIWrapper
from langchain.agents import (AgentExecutor, AgentOutputParser,
                              LLMSingleActionAgent, Tool)
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.utilities import GoogleSearchAPIWrapper, TextRequestsWrapper

from tools import (extract_text, extract_texts_from_urls,
                   ask_open_ai, get_tools, tools)
import re

llm = OpenAI(temperature=0, verbose=True)
search = GoogleSearchAPIWrapper()
request = TextRequestsWrapper()

template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Action Summary: a summary of the action
Observation: the result of the action
... (this Thought/Action/Action Input/Action Summary/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! 

Question: {input}
{agent_scratchpad}"""

class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools_getter: Callable
    
    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        tools = self.tools_getter(kwargs["input"])
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        return self.template.format(**kwargs)

prompt = CustomPromptTemplate(
    template=template,
    tools_getter=get_tools,
    input_variables=["input", "intermediate_steps"]
)

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        action_input = re.sub(r'\n.*$', '', action_input).strip()  # Remove the extra data from the input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)



class Assistant:
    def __init__(self):
        
        self.llm = OpenAI(temperature=0)
        self.llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        tool_names = [tool.name for tool in tools]
        self.agent = LLMSingleActionAgent(
            llm_chain=self.llm_chain,
            output_parser=CustomOutputParser(),
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )
        self.agent_executor = AgentExecutor.from_agent_and_tools(agent=self.agent, tools=tools, verbose=True)

    def execute(self, query: str):
        self.agent_executor.run(query)