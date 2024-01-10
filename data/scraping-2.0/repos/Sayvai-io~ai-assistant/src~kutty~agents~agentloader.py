# Agent Loader Class
# Loads the agent from the agent file

from typing import List, Any
from rich import print as rprint
from langchain.agents import initialize_agent

class AgentLoader:
    """This class is used to load Agent"""
    
    def __init__(self, tools: Any, llm: Any, agent_name: str, verbose: bool = False):
        """Initializes the AgentLoader class"""
        self.tools = tools
        self.agent = None
        self.llm = llm
        self.agent = initialize_agent(self.tools, self.llm, agent_name, verbose)  
        rprint(f"[bold green]Agent {agent_name} loaded successfully![/bold green]")
    
    def get_agent(self) -> Any:
        """Returns the agent"""
        return self.agent 
    
    def get_agent_template(self) -> List[str]:
        """Returns the agent template"""
        return self.agent.agent.llm_chain.prompt.template
    
    
        