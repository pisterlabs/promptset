import json
from tabnanny import verbose
from pydantic import Field, BaseModel
from langchain.tools import BaseTool
from typing import Type
from ..schema.weather import GetCurrentWeatherInput, WeatherModel
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI

def get_current_wether(location: str) -> WeatherModel:
    wether_info = {
            "location": location,
            "temperature": 20.0,
            "unit": "celsius",
            "forecast": "sunny",
        }

    return json.dumps(wether_info)

class GetCurrentWeatherTool(BaseTool):
    name = 'get_current_wether'
    description = '現在の天気を知ることができます'

    args_schema: Type[BaseModel] = GetCurrentWeatherInput

    def _run(self, location: str) -> WeatherModel:
        current_wether = get_current_wether(location)
        return current_wether

    def _arun(self, location: str):
        raise NotImplementedError("This tool does not support async")

def execute_agent(request: str, debug: bool = False):
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    tools = [GetCurrentWeatherTool()]
    agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=debug)
    result = agent.run(request)
    return result



