
import os
from sayvai_tools.tools.sql_database import Database
from sayvai_tools.tools.conversational_human import ConversationalHuman as human
from sayvai_tools.tools.calendar import Calendar
from constants import PROMPT, LLM
# from langchain.tools import HumanInputRun as human
from langchain.agents import AgentType, Tool, AgentExecutor , initialize_agent , OpenAIFunctionsAgent
from sqlalchemy import create_engine
from langchain.memory import ConversationSummaryBufferMemory


with open("openai_api_key.txt", "r") as f:
    api_key = f.read()
    

os.environ["OPENAI_API_KEY"] = api_key


llm = LLM

class Assistant:
    """
    The assistant is a class that is used to interact with the user and the agent. 
    It is the main interface for the user to interact with the agent."""
    def __init__(self):
        self.agent = None
        self.memory = ConversationSummaryBufferMemory(llm=llm)
        self.tools = None
        self.human = None
        self.sql = None
        self.voice = None
        self.calendly = None
        self.system_message = PROMPT
        self.prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=self.system_message,
        )
    
    
    def initialize_human(self) -> None:
        """Initialize the human"""
        self.human = human()
        return None
 
    def initialize_tools(self):
        """Initialize the tools"""
        if self.tools is None:
            raise ValueError("Tools not initialized")
        else :
            print("Tools already initialized")
            

            
    def agent_inittialize(self, verbose: bool = False) -> None:
        """Initialize the agent"""
        self.agent = OpenAIFunctionsAgent(
            llm=llm,
            prompt=self.prompt,
        )
        agent_executor =AgentExecutor(
            agent=self.agent,
            verbose=verbose,
            memory=self.memory,
            max_iterations=30
        )
        return agent_executor
        
        
     
    def initialize(self, verbose: bool=False) -> None:
        """Initialize the assistant"""
        # self.initialize_vectordb()
        # self.initialize_tools()
        self.agent_executor = self.agent_inittialize(verbose=verbose)
        return None
    
    def get_answer(self, query:str) -> str:
        """Get the answer from the agent"""
        return self.agent_executor.run(query)