# To replace human agent with your own agent, rename this file to myagent.py and replace the class name with MyAgent.
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from typing import List
from langchain import LLMChain
from agents.customprompt.prompt import CustomPromptTemplate
from agents.parser.parser import CustomOutputParser
from agents.templates import template, template_with_history
from langchain.memory import ConversationBufferWindowMemory


class CustomHumanAgent:
    """Custom agent """
    def __init__(self,llm, tools : List[Tool]) -> None:
        self.prompt = CustomPromptTemplate(
            template= template_with_history, 
            tools=tools,  
            input_variables=["input", "intermediate_steps", "history"])  
        self.llm_chain = LLMChain(llm=llm, prompt=self.prompt)
        self.output_parser = CustomOutputParser()
        self.agent_executor = None
        self.tools = tools
        pass
    
    def create_agent(self, verbose: bool = True) -> None:
        """Create agent"""
        tool_names = [tool.name for tool in self.tools]
        agent = LLMSingleActionAgent(
            llm_chain=self.llm_chain,
            output_parser=self.output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )
        # print(agent)
        # print(tool_names)
        memory=ConversationBufferWindowMemory(k=2)
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent, 
            tools=self.tools, 
            verbose=verbose ,
            memory=memory
            )
        pass
        
    
    def run(self, input: str) -> str:
        """Run agent"""
        return self.agent_executor.run(input=input)