import traceback
import config

from datetime import date, timedelta
from common.rabbit_comms import publish, publish_action, publish_actions
from common.utils import tool_description, tool_error

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from O365 import Account, FileSystemTokenBackend, MSGraphProtocol
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.tools import BaseTool
from langchain.tools import StructuredTool
from typing import Any, Dict, Optional, Type
import ast


def generate_commands(text):
    
    chat = ChatOpenAI(temperature=0, model_name="gpt-4.0")

    query = f""" {text}. Suggest further actions the human should consider as a list of dicts format "title", "action" 
    Example: "[('Create Email', 'Create an email with the following: text'),('Create Task', 'Create a task with the following: text')]"""

    print(f"Function Name: generate_commands | Query: {query}, Text: {text}")
    response = chat([HumanMessage(content=query)]).content
    return ast.literal_eval(response)

class Help(BaseTool):
    parameters = []
    optional_parameters = []
    name = "HELP"
    summary = """Useful for when you want details on a tool """
    parameters.append({"name": "tool_name", "description": "Tool to return help on" })
    description = tool_description(name, summary, parameters, optional_parameters)
    return_direct = True
    
    tools = []
    def init(self, tools):
        self.tools = tools
    

    def _run(self, tool_name: str, publish: str = "True", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        
        try:
            for tool in self.tools:
                if tool.name.lower() == tool_name.lower():
                    if publish.lower() == "true":
                        publish(tool.description)
                        return config.PROMPT_PUBLISH_TRUE
                    else:
                        return(tool.description)

            raise Exception(f"Could not find Tool {tool_name}")

        except Exception as e:
            traceback.print_exc()
            return tool_error(e, self.description)
        

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("HELP does not support async")