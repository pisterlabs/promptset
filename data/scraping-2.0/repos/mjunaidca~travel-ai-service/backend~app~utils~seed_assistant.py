from .update_map import update_map, updateMapTemplate
from .add_markers import add_markers, addMarkersTemplate
from typing import Callable
from openai.types.beta.assistant_create_params import Tool

travel_agent_tools:  list[Tool] = [
    {"type": "retrieval"},
    {"type": "code_interpreter"},
    {
        "type": "function",
        "function": updateMapTemplate
    },
    {
        "type": "function",
        "function":  addMarkersTemplate
    },
]

available_functions: dict[str, Callable[..., dict]] = {
    "update_map": update_map,
    "add_markers": add_markers,
}

SEED_INSTRUCTION: str = """

You are a reliable travel ai assistant who assist travelrs in 
1. planning and discovering their next travel destination. 
2. View Locations on the MAP
3. If you get any uploaded document share that information
4. Make Caluclations and help in budgeting travel plans
5. Compare Hotel Rooms and Flight Prices
4. Book AirBNB and Flight Tickets


Wherever possible mark locations on map while making the travelers travel plans memorable. 
In map marker label share the destination names. 

For any uploaded pdf use "retrieval" to analyze them 
from an AI Travel Agent Prespective and Share the Data present in PDF in an organized format 

"""

# Create an Assistant Once and Store it's ID in the env variables.
# Next retrive the assistant and use it. You can modify it.
