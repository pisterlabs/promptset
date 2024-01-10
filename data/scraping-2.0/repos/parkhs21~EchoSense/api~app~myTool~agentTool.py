from langchain import hub

from langchain_core.pydantic_v1 import BaseModel, Field, validator

from langchain.agents import AgentType, initialize_agent, load_tools, AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool, Tool, StructuredTool
from langchain.agents.agent_toolkits.amadeus.toolkit import AmadeusToolkit
from langchain.schema.runnable import RunnableLambda

import os
import json
import requests
import dotenv
dotenv.load_dotenv()

llm = ChatOpenAI(model="gpt-4-1106-preview")


@tool
def search_near_place_gps(data: str) -> str:
    "input data = 'latitude, longitude, place_type'"
    "Tool for finding places around a current gps location."
    "We're going to find places around a current gps location. What place_type of places will we find?"
    "latitude, Longitude is the location to search around."
    "plcae_type is the type of place to search for."
    """You should be able to recognize and interpret a user's needs, even if they are expressed in an indirect or direct way.
    For example, if a user says, "Recommend a restaurant," you should recommend a restaurant based on the user's location.
    If a user says they are "tired" or "exhausted" or mentions traveling a long distance, you should interpret this as a request for accommodation options such as a hotel.
    If the user's request is unclear or has multiple interpretations, ask for further clarification.
    """
    latitude, longitude, place_type = data.split(",")
    location = f"{latitude}, {longitude}"
    radius = 1500
    API_KEY = os.getenv("GOOGLE_API_KEY")
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={location}&radius={radius}&type={place_type}&language=ko&key={API_KEY}"
    response = json.loads(requests.get(url).content)["results"]

    places = []
    for place in response[:2]:
        places.append({
            "name": place["name"],
            "place_id": place["place_id"]
        })
    return json.dumps(places)


@tool
def search_near_place_str(data: str) -> str:
    "input data = 'location, place_type'"
    "Tool for finding places around a specific location."
    "We're going to find places around a specific location. What place_type of places will we find?"
    "location is the name of the place I want to find."
    "plcae_type is the type of place to search for."
    """You should be able to recognize and interpret a user's needs, even if they are expressed in an indirect or direct way.
    For example, if a user says, "Recommend a restaurant," you should recommend a restaurant based on the user's location.
    If a user says they are "tired" or "exhausted" or mentions traveling a long distance, you should interpret this as a request for accommodation options such as a hotel.
    If the user's request is unclear or has multiple interpretations, ask for further clarification.
    """
    location, place_type = data.split(",")
    radius = 1500
    API_KEY = os.getenv("GOOGLE_API_KEY")
    url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={location}&radius={radius}&type={place_type}&language=ko&key={API_KEY}"
    response = json.loads(requests.get(url).content)["results"]

    places = []
    for place in response[:2]:
        places.append({
            "name": place["name"],
            "place_id": place["place_id"]
        })
    return json.dumps(places)


@tool
def search_place_info(place_id: str) -> str:
    "Tool for finding information about a place."
    "You want to get information about a specific store."
    "place_id is the ID of the place to search for."

    API_KEY = os.getenv("GOOGLE_API_KEY")
    url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&language=ko&key={API_KEY}"
    response = json.loads(requests.get(url).content)['result']

    place_info = {}
    place_info["name"] = response.get("name","N/A")
    place_info["rating"] = response.get("rating","N/A")
    place_info["user_ratings_total"] = response.get("user_ratings_total","N/A")
    place_info["vicinity"] = response.get("vicinity","N/A")

    return json.dumps(place_info)


tools = [search_near_place_gps, search_near_place_str, search_place_info]
# toolkit = AmadeusToolkit()
# for t in toolkit.get_tools():
#     tools.append(t)

prompt = hub.pull("hwchase17/react")
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

llm_with_stop = llm.bind(stop=["\nObservation"])

agent = (
    {
        "input": lambda x: f"system_addition_info: {x['system']}\n" + 
                            f"prompt_human_wanted_input: {x['input']}",
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"])
    }
    | prompt
    | llm_with_stop
    | ReActSingleInputOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools)
agent_executor_withsteps = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True, handle_parsing_errors=True)
# addition_info = {
#     "latitude": 37.59552,
#     "longitude": 126.87083,
#     "current_time": '2023-12-06T16:00:00'
# }
# agent_executor.invoke(
#     {
#         "system": json.dumps(addition_info),
#         "input": "내일(2023년 12월 7일) 제주도 가는 항공편 찾아서 출발시간, 도착시간 등 항공권 정보 알려줘."
#     }
# )