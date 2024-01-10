from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

class Validation(BaseModel):
    plan_is_valid: str = Field(
        description="This field is 'yes' if the plan is feasible, 'no' otherwise"
    )
    updated_request: str = Field(description="Your update to the plan")


class ValidationTemplate(object):
    def __init__(self):
        self.system_template = """
    You are a travel agent who helps users make exciting travel plans.

    The user's request will be denoted by four hashtags. Determine if the user's
    request is reasonable and achievable within the constraints they set.

    A valid request should contain the following:
    - A start and end location
    - A trip duration that is reasonable given the start and end location
    - Some other details, like the user's interests and/or preferred mode of transport

    Any request that contains potentially harmful activities is not valid, regardless of what
    other details are provided.

    If the request is not valid, set
    plan_is_valid = 0 and use your travel expertise to update the request to make it valid,
    keeping your revised request shorter than 100 words.

    If the request seems reasonable, then set plan_is_valid = 1 and
    don't revise the request.

    {format_instructions}
    """

        self.human_template = """
    ####{query}####
    """

        self.parser = PydanticOutputParser(pydantic_object=Validation)

        self.system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.system_template,
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(
            self.human_template, input_variables=["query"]
        )

        self.chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_message_prompt, self.human_message_prompt]
        )

import requests
from pathlib import Path

def search_restaurant_location_ID(location):

    url =  "https://tripadvisor16.p.rapidapi.com/api/v1/restaurant/searchLocation"

    querystring = {"query":location}

    headers = {
        "X-RapidAPI-Key": "53bd119ccfmsh364f7fc48f6cb7bp182915jsnd2899197f369",
        "X-RapidAPI-Host": "tripadvisor16.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    json = response.json()
    id = str(json['data'][0]['locationId']) 

    return id

def search_restaurant(location_id): 
    import requests

    url = "https://tripadvisor16.p.rapidapi.com/api/v1/restaurant/searchRestaurants"
    querystring = {"locationId":location_id}

    headers = {
        "X-RapidAPI-Key": "53bd119ccfmsh364f7fc48f6cb7bp182915jsnd2899197f369",
        "X-RapidAPI-Host": "tripadvisor16.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    restaurants = response.json()
    
    restaurant_list = {}
    for i in restaurants['data']['data']:  
        restaurant_list[i['name']] = [i['establishmentTypeAndCuisineTags'], i['priceTag'], i['averageRating']]

    return restaurant_list


class ItineraryTemplate_v2(object):
    def __init__(self):
        self.system_template = """
        You are a travel agent who helps users make exciting travel plans.

        The user's request will be denoted by four hashtags. Convert the user's request into a fun, detailed itinerary describing the city they should visit, restaurants they should go to for breakfast, lunch, and dinner, and the activities they should do.

        Try to include the specific address of each location.

        Remember to take the user's preferences and timeframe into account, and give them an itinerary that would be fun and doable given their constraints.

        Return the itinerary as a dictionary of dictionaries in the following format:
        (Put curly braces around the whole thing and around the value for each day)

        day1: itinerary: text, city: city, country: , breakast: breakast, lunch: lunch, dinner: dinner acitivity: activity,

        day2: itinerary: text, city: city, country: , breakfast: breakast, lunch:lunch--, dinner: dinner, acitivity: activity
        

        For the itinerary value, input a detailed itinerary. 

        Stay in only one city the  whole day.
        The city, breakfast, lunch, dinner, and activity values should all come from the itinerary.
        Return the dictionary and nothing else. 

        Only use the breakfast data from this. 

        

    """

        self.human_template = """
    ####{query}####
    """

        self.system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.system_template,
        )
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(
            self.human_template, input_variables=["query"]
        )

        self.chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_message_prompt, self.human_message_prompt]
        )

class UpdateTemplate(object):
    def __init__(self):
        self.system_template = """
        You are a travel agent who helps users make exciting travel plans.

        Update the itinerary below based on the updated query requirements.
        Don't change the keys of the dictionary, just the values of it. 

        {itinerary}
    """

        self.human_template = """
    ####{update_query}####
    """

        self.system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.system_template,

        )
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(
            self.human_template, input_variables=["query"]
        )

        self.chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_message_prompt, self.human_message_prompt]
        )


class Trip(BaseModel):
    start: str = Field(description="start location of trip")
    end: str = Field(description="end location of trip")
    waypoints: List[str] = Field(description="list of waypoints")
    transit: str = Field(description="mode of transportation")

class MappingTemplate(object):
    def __init__(self):
        self.system_template = """
      You an agent who converts detailed travel plans into a simple list of locations.

      The itinerary will be denoted by four hashtags. Convert it into
      list of places that they should visit. Try to include the specific address of each location.

      Your output should always contain the start and end point of the trip, and may also include a list
      of waypoints. It should also include a mode of transit. The number of waypoints cannot exceed 20.
      If you can't infer the mode of transit, make a best guess given the trip location.

      For example:

      ####
      Itinerary for a 2-day driving trip within London:
      - Day 1:
        - Start at Buckingham Palace (The Mall, London SW1A 1AA)
        - Visit the Tower of London (Tower Hill, London EC3N 4AB)
        - Explore the British Museum (Great Russell St, Bloomsbury, London WC1B 3DG)
        - Enjoy shopping at Oxford Street (Oxford St, London W1C 1JN)
        - End the day at Covent Garden (Covent Garden, London WC2E 8RF)
      - Day 2:
        - Start at Westminster Abbey (20 Deans Yd, Westminster, London SW1P 3PA)
        - Visit the Churchill War Rooms (Clive Steps, King Charles St, London SW1A 2AQ)
        - Explore the Natural History Museum (Cromwell Rd, Kensington, London SW7 5BD)
        - End the trip at the Tower Bridge (Tower Bridge Rd, London SE1 2UP)
      #####

      Output:
      Start: Buckingham Palace, The Mall, London SW1A 1AA
      End: Tower Bridge, Tower Bridge Rd, London SE1 2UP
      Waypoints: ["Tower of London, Tower Hill, London EC3N 4AB", "British Museum, Great Russell St, Bloomsbury, London WC1B 3DG", "Oxford St, London W1C 1JN", "Covent Garden, London WC2E 8RF","Westminster, London SW1A 0AA", "St. James's Park, London", "Natural History Museum, Cromwell Rd, Kensington, London SW7 5BD"]
      Transit: driving

      Transit can be only one of the following options: "driving", "train", "bus" or "flight".

      {format_instructions}
    """

        self.human_template = """
      ####{agent_suggestion}####
    """

        self.parser = PydanticOutputParser(pydantic_object=Trip)

        self.system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.system_template,
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(
            self.human_template, input_variables=["agent_suggestion"]
        )

        self.chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_message_prompt, self.human_message_prompt]
        )