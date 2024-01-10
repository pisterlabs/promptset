from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import Tool
from langchain.chains import LLMChain, LLMMathChain
from langchain_experimental.utilities import PythonREPL
from chatglm3_6b_llm import Chatglm3_6b_LLM
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)

#指定ChatGLM2-6B的API endpoint url，用langchain的ChatOpanAI类初始化一个ChatGLM的chat模型
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(
        model_name="chatglm",
        openai_api_base="http://127.0.0.1:6006/v1",
        openai_api_key="1234",
        streaming=False,
    )

class WeatherInput(BaseModel):
    city_name: str = Field(description="城市")

@tool("get_weather_tool", return_direct=True, args_schema=WeatherInput)
def get_weather_tool(city_name: str) -> str:
    """ get_weather_tool 根据城市获取当地的天气"""
    print(f"get_weather_tool the current weather for:{city_name}")
    return str({"city":city_name,"weather":"多云 23°C"})


tools = [get_weather_tool]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,
                         handle_parsing_errors=True, max_iterations=3, early_stopping_method="generate")

user_input = "杭州的天气如何？"
answer = agent.run(user_input)
print(answer)
