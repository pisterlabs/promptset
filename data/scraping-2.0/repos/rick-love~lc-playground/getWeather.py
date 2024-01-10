# This file is used to get the weather for a given location
from config import get_OpenAI, get_OpenWeatherAPIKey
from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools


# Set the API key for OpenAI
try:
    OpenAI.api_key = get_OpenAI()
except Exception as e:
    raise Exception(f"Error setting API key for OpenAI: {e}")

# Set the API key for OpenWeatherMap
try:
    openWeather_api_key = get_OpenWeatherAPIKey()
except Exception as e:
    raise Exception(f"Error setting API key for OpenAI: {e}")

llm = OpenAI(temperature=0)
tools = load_tools(["openweathermap-api"], llm)

agent_chain = initialize_agent(tools=tools, llm=llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent_chain.run("What is the weather in New York?")