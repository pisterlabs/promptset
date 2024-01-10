""" Using Langchain Going to Generate a Server that can be used to shcedule meetings """
# Importing the required libraries
import os
from langchain.chat_models import ChatOpenAI
from typing import List, Dict, Any
from langchain.agents import load_tools
from langchain.agents import initialize_agent
# from langchain.llms import OpenAI


with open("../../openai_api_key.txt", "r") as f:
    OPEN_AI_KEY = f.read()

with open("../../serp_api_key.txt", "r") as f:
    SERP_API_KEY = f.read()

os.environ["OPENAI_API_KEY"] = OPEN_AI_KEY
os.environ["SERPAPI_API_KEY"] = SERP_API_KEY

class LlmServer:
    """LLm server class"""
    def __init__(self) -> None:
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
        self.tools = None
        self.agent = None     
        
    
    def initialize(self, tools: List, agent: str) -> str:
        """Initialize the server"""
        if self.tools is None:
            self.tools = load_tools(tools, llm=self.llm)
        self.agent = initialize_agent(self.tools, self.llm, agent, verbose=False)
        return "Initialized"
    
    def get_response(self, text: str) -> Dict[str, Any]:
        response = self.agent.run(text)
        return response
    



