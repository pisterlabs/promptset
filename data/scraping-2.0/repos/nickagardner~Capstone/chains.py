from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import BaseOutputParser
from langchain.chains import LLMChain

import re

class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""


    def parse(self, text: str):
        """Parse the output of an LLM call."""
        return text.strip().split(", ")

def choose_func(text, llm):
    template = """You are a helpful assistant who parses text and determines what change the user is requesting. 
    A user will pass in text, which you should parse to determine which change is requested. 
    Return ONLY the response to the last user request and nothing more.

    Change options are as follows:
        1. avoid_area - add if the user requests to avoid a particular area, or to not take a particular route.
        2. add_waypoints - add if the user requests to add additional stops to their route, or if they want to route through a destination.
        3. prefer_path_type - add if the user specifies a type of path surface that is preferred.
    
    User: prefer trails
    Assistant: prefer_path_type | trails

    User: stop at boston common
    Assistant: add_waypoints | boston common

    User: avoid main street
    Assistant: avoid_area | main street

    User: route through north ave and city hall
    Assistant: add_waypoints | north ave | city hall

    User: avoid johnson bridge and 17 Madison St
    Assistant: avoid_area | johnson bridge | 17 Madison St

    User: I want to ride on roads
    Assistant: prefer_path_type | roads

    User: don't go on fipson dr
    Assistant: avoid_area | fipson dr

    User: stick to bike lanes
    Assistant: prefer_path_type | bike lanes

    User: add 1050 wilkins dr and jackson library to stops
    Assistant: add_waypoints | 1050 wilkins dr | jackson library
    
    User: """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}\nAssistant:"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chain = LLMChain(
        llm=llm,
        prompt=chat_prompt,
    )

    print(f"Choose input: {text}")
    result = "Assistant: " + chain.run(text)
    result = result + "\n"

    print(f"Choose result: {result}")

    pattern = ".*Assistant:.*\n"
    match = re.search(pattern, result)[0]

    content = match.split("Assistant:")[-1]
    components = content.split("|")
    function = components[0].strip()
    parameters = []
    for parameter in components[1:]:
        parameters.append(parameter.strip("\"\',.`\n "))

    return function, parameters

def split_changes(text, llm):
    template = """You are a helpful assistant who parses text and splits the text into discrete requests.
    Return ONLY the response to the last user response and nothing more.
    
    User: I want to ride on trails and avoid main street
    Assistant: I want to ride on trails | avoid main street

    User: route through boston common and the empire state building
    Assistant: route through boston common and the empire state building

    User: avoid 42nd ave
    Assistant: avoid 42nd ave

    User: stop at 120 charles rd and the golden gate bridge and prefer roads
    Assistant: stop at 120 charles rd and the golden gate bridge | prefer roads

    User: avoid johnson bridge and 17 Madison St
    Assistant: avoid johnson bridge and 17 Madison St

    User: pass through the airport and use city streets
    Assistant: pass through the airport | use city streets

    User: skip 12th st and stop at Shelly McFarlin Park and the Natural History Museum.
    Assistant: skip 12th st | stop at Shelly McFarlin Park and the Natural History Museum.

    User: stop at the park
    Assistant: stop at the park

    User: add 1050 wilkins dr and jackson library to stops, stay away from west ham
    Assistant: add 1050 wilkins dr and jackson library to stops | stay away from west ham

    User: I want to ride on roads
    Assistant: I want to ride on roads

    User: don't go on fipson dr and preferred trails
    Assistant: don't go on fipson dr | preferred trails

    User: skip welter ave
    Assistant: skip welter ave
    
    User: """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}\nAssistant:"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chain = LLMChain(
        llm=llm,
        prompt=chat_prompt,
        output_parser=CommaSeparatedListOutputParser()
    )
    print(f"Split input: {text}")
    result = "Assistant: " + chain.run(text)[0]
    result = result + "\n"

    print(f"Split result: {result}")

    pattern = ".*Assistant:.*\n"
    match = re.search(pattern, result)[0]

    content = match.split("Assistant:")[-1]
    components = content.split("|")
    requests = []
    for parameter in components:
        requests.append(parameter.strip("\"\',.`\n "))

    return requests

def mod_or_trail(text, llm):
    template = """You are a helpful assistant who parses text and determines whether the user text is a modification request or a request for information on nearby trails.
    A user will pass in text, which you should parse to determine which type of request is being made.
    Return ONLY the response to the last user request and nothing more.

    Request options are as follows:
        1. Modification - add if the user requests to modify their route.
        2. Trail - add if the user requests information on nearby trails.
    
    User: what parks are good for biking
    Assistant: Trail

    User: list nearby trails
    Assistant: Trail

    User: stop at boston common
    Assistant: Modification

    User: route through north ave and city hall
    Assistant: Modification
    
    User: show me Nature Park routes
    Assistant: Trail

    User: avoid johnson bridge and 17 Madison St
    Assistant: Modification

    User: Looking for intermediate trails with rating above 3
    Assistant: Trail

    User: prefer trails
    Assistant: Modification
    
    User: """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}\nAssistant:"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chain = LLMChain(
        llm=llm,
        prompt=chat_prompt,
    )

    print(f"Mod/Trail input: {text}")
    result = "Assistant: " + chain.run(text)
    result = result + "\n"

    print(f"Mod/Trail result: {result}")

    pattern = ".*Assistant:.*\n"
    match = re.search(pattern, result)[0]

    content = match.split("Assistant:")[-1]
    type = content.strip("\"\',.`\n ")

    return type

def extract_trail_info(text, llm):
    template = """You are a helpful assistant who parses text and extracts user requirements from their request.
    A user will pass in text, which you should parse to determine which requirements the user specifies.
    Return ONLY the response to the last user request and nothing more.

    Request options are as follows:
        1. Difficulty - difficulty rating of the trail. Options here are easy, intermediate, and hard.
        2. Distance - how far from the user's current location the trail is.
        3. Rating - rating of the trail. Options here are floats between 0 and 5.
        4. Length - length of the trail.

    User: list nearby trails
    Assistant: No changes

    User: Looking for intermediate trails with rating above 3
    Assistant: Difficulty(Intermediate) | Rating(> 3)

    User: show me Nature Park routes
    Assistant: No changes

    User: routes with rating above 2 and length of 5 miles or less
    Assistant: Rating(> 2) | Length(< 5)

    User: what are good places to ride nearby
    Assistant: No changes

    User: what trails are within 10 miles and are easier than advanced
    Assistant: Distance(< 10) | Difficulty(< Advanced)

    User: Looking for easy trails
    Assistant: Difficulty(Easy)

    User: tell me about trails nearby
    Assistant: No changes
    
    User: """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}\nAssistant:"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chain = LLMChain(
        llm=llm,
        prompt=chat_prompt,
    )

    print(f"Trail Extract input: {text}")
    result = "Assistant: " + chain.run(text)

    print(f"Trail Extract result: {result}")
    result = result + "\n"

    pattern = ".*Assistant:.*\n"
    match = re.search(pattern, result)[0]
    content = match.split("Assistant:")[-1]

    if "No changes" in content:
        return None
    else:
        components = content.split("|")
        requests = {}
        for request in components:
            type = request.split("(")[0].strip("\"\',.`\n ")
            value = request.split("(")[1].strip("\"\',.`\n )")
            if len(value.split(" ")) > 1:
                operator, value = value.split(" ")
                requests[type] = [value, operator]
            else:
                requests[type] = [value]

        return requests

