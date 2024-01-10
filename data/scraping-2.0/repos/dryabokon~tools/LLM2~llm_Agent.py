import re
from typing import List, Union
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain.agents import AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.agents import initialize_agent, AgentType, Tool
# ----------------------------------------------------------------------------------------------------------------------
import tools_time_profiler
# ----------------------------------------------------------------------------------------------------------------------
class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(return_values={"output": llm_output.split("Final Answer:")[-1].strip()},log=llm_output)

        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        tool_input = match.group(2).strip(" ").strip('"')
        return AgentAction(tool=action, tool_input=tool_input, log=llm_output)
# ----------------------------------------------------------------------------------------------------------------------
class CustomPromptTemplate(StringPromptTemplate):

    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)
# ----------------------------------------------------------------------------------------------------------------------
prompt_template_str = """Answer the following questions, You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat 3 times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Previous conversation history:
{history}

Question: {input}
{agent_scratchpad}
"""
# ----------------------------------------------------------------------------------------------------------------------
class Agent(object):
    def __init__(self,LLM,tools,verbose=True):
        self.TP = tools_time_profiler.Time_Profiler()
        self.LLM = LLM
        self.tools = tools
        #self.agent = self.init_agent(agent_type=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,verbose=verbose)
        self.agent = self.init_agent_custom(verbose)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def run_query(self, query):
        responce = self.agent.run(query)
        return responce, []
# ----------------------------------------------------------------------------------------------------------------------
    def init_agent(self,agent_type,verbose=True):
        memory = ConversationBufferWindowMemory(memory_key='chat_history', k=5, return_messages=True)
        agent_executor = initialize_agent(self.tools, self.LLM, agent=agent_type,memory=memory,verbose=verbose,return_intermediate_steps=False)
        return agent_executor
# ----------------------------------------------------------------------------------------------------------------------
    def init_agent_custom(self,verbose=True):
        # agent_executor0 = initialize_agent(self.tools, self.LLM,agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=verbose,return_intermediate_steps=True)
        # prompt_template_str0 = agent_executor0.agent.llm_chain.prompt.messages[0].prompt.template
        prompt_template = CustomPromptTemplate(template=prompt_template_str,tools=self.tools,input_variables=["input", "intermediate_steps","history"])
        #custom_agent.llm_chain.run({'input':'1+1','intermediate_steps':[]})

        custom_agent = LLMSingleActionAgent(
            llm_chain=LLMChain(llm=self.LLM, prompt=prompt_template),
            output_parser=CustomOutputParser(),
            stop=["\nObservation:"],
            allowed_tools=[t.name for t in self.tools]
        )
        memory = ConversationBufferWindowMemory(k=2)

        agent_executor = AgentExecutor.from_agent_and_tools(agent=custom_agent, tools=self.tools, verbose=verbose,memory=memory,intermediate_steps=True,handle_parsing_errors=True)
        return agent_executor
# ----------------------------------------------------------------------------------------------------------------------
