#Tools file for langchain
# from pydantic import BaseModel
from typing import List
from langchain.agents import load_tools

class ToolLoader:
    
    """ This class is used to load the tools """
    def __init__(self, tools: list, llm) -> None:
        """ This is the constructor of the class """
        if not isinstance(tools, list):
            raise TypeError("tools must be a list")
        if len(tools) == 0:
            raise ValueError("tools must not be empty")
        self.tools = load_tools(tools, llm=llm)
        # print(self.tools)
        # return self.tools
    
    def get_tools(self):
        """ This method is used to get the tools """
        return self.tools
        
    def get_descriptions(self):
        """ This method is used to get the descriptions of the tools """
        return [tool.description for tool in self.tools]
    
    def get_names(self):
        """ This method is used to get the names of the tools """
        return [tool.name for tool in self.tools]