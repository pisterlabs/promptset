import os, requests, datetime, pytz
from typing import Optional
import urllib.parse
from dotenv import load_dotenv

from llama_index.agent import OpenAIAssistantAgent
from llama_index.tools import FunctionTool
from llama_index.readers.schema.base import Document
from llama_index.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.tool_spec.load_and_search.base import LoadAndSearchToolSpec
# from llama_index.readers.web import BeautifulSoupWebReader

from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
import json


def main(user_message, api_key, email):
    load_dotenv()
    # Custom Google Search Engine / Deployment ->
    #   Could be used for all users but only 10k requests are free per day.
    Custom_Search_API_KEY = os.getenv("Custom_Search_API_KEY") # get this from github codespace secrets
    Custom_Search_Engine_ID = os.getenv("Custom_Search_Engine_ID") # get this from github codespace secrets

    # User's OpenAI API Key
    # print("api_key: ", api_key)
    os.environ["OPENAI_API_KEY"] = api_key

    currentDateTimeInCity_tool = FunctionTool.from_defaults(fn=currentDateTimeInCity)
    mostFreeDays_tool = FunctionTool.from_defaults(fn=get_free_time)

    eventInfo_tools = EventInfoToolSpec(key=Custom_Search_API_KEY, engine=Custom_Search_Engine_ID)
    formatted_event_info_tool = FunctionTool.from_defaults(fn=eventInfo_tools.formatted_event_info)

    gsearch_tools = GoogleSearchToolSpec(key=Custom_Search_API_KEY, engine=Custom_Search_Engine_ID).to_tool_list()
    gsearch_load_and_search_tools = LoadAndSearchToolSpec.from_defaults(gsearch_tools[0]).to_tool_list()

    all_tools = [currentDateTimeInCity_tool, formatted_event_info_tool, *gsearch_load_and_search_tools, *gsearch_tools[1::], mostFreeDays_tool]
    
    agent = OpenAIAssistantAgent.from_new(
        name="Exploration Assistant",
        model="gpt-4-1106-preview",
        instructions=\
        "You are a helpful assistant that finds interesting activities and upcoming events for users\
        based on their interests, location, local time, most free days, etc. If you are not sure what city the user would like to explore,\
        ask them first and use the tool provided to determine the local date and time in the city.\
        When you recommend events for the user to attend in the city of interest, make sure to only search for days when the user is most free. \
        Make sure to include both ongoing and future events, but not events that have ended. \
        Make sure you include the urls to the events.",
        tools=all_tools,
        # instructions_prefix="Please address the user as Jerry.",
        verbose=True,
        # run_retrieve_sleep_time=1.0,
    )
    
    print("assitant created: ", agent._assistant.id)
    agent.add_message(f"My email is {email}")

    response = agent.chat(user_message)
    return response.response

def get_free_time(email, scope: int = 7):
    """
    Retrieves the free time slots for a given email address within a specified scope.

    Parameters:
    - email (str): The email address for which to retrieve free time slots.
    - scope (int): The number of days to consider for free time slots. Between 0 and 30. Default is 7.

    Returns:
    - list: A list of dictionaries containing the date and free time for each slot.
    """
    
    url = "http://localhost:3000/free-slot"
    headers = {'Content-Type': 'application/json'}

    data = {
        "orgId": "1",
        "email": email,
        "scope": scope
    }

    response = requests.get(url, headers=headers, data=json.dumps(data))
    slots = response.json().get("freeSlots", [])
    return slots

def cityTimeZone(city_name:str):
  """Get the time zone for the city of interest"""
  geolocator = Nominatim(user_agent="timezone_app")
  # Use geopy to get the coordinates (latitude and longitude) of the city
  location = geolocator.geocode(city_name)

  output = None
  if location:
      latitude = location.latitude
      longitude = location.longitude
      tz_finder = TimezoneFinder()

      # Get the time zone of the specified coordinates
      timezone_str = tz_finder.timezone_at(lat=latitude, lng=longitude)
      if timezone_str:
          # print(f"Time zone of {city_name}: {timezone_str}")
          output = timezone_str
      else:
          print("Time zone not found for the provided coordinates.")
  else:
      print(f"Coordinates not found for {city_name}.")
  return output

def currentDateTimeInTimeZone(timeZone:str):
  """Get the current date and time in a provided time zone"""
  desired_timezone = pytz.timezone(timeZone)
  current_time = datetime.datetime.now(desired_timezone)
  current_time_str = current_time.isoformat()
  # print(current_time_str)
  return current_time_str

def currentDateTimeInCity(city_name:str):
  """Get the current date and time in a city of interest"""
  localTimeZone = cityTimeZone(city_name)
  now = currentDateTimeInTimeZone(localTimeZone)
  output = f"The city to explore is {city_name}\n" + \
           f"The time zone of {city_name} is {localTimeZone}\n" + \
           f"Current date and time is {now}\n"
  return output

def process_rich_snippets(rich_snippets):
    processed_data = []
    # Processing for event information
    if 'Event' in rich_snippets:
        for event in rich_snippets['Event']:
            e = {}
            e['event_name'] = event.get('name')
            e['event_start_date'] = event.get('startDate')
            # e['event_end_date'] = event.get('endDate')
            e['event_description'] = event.get('description')
            e['event_link'] = event.get('url')
            processed_data.append(e)
    return processed_data

def event_dict_to_paragraph(event_dict):
    """
    Converts an event dictionary to a descriptive paragraph in natural language.
    """
    paragraph = ""
    name = event_dict.get('event_name', 'This event')
    if event_dict.get('event_start_date'):
        date_range = f"on {event_dict['event_start_date']}"

    description = event_dict.get('event_description', "")
    link = event_dict.get('event_link', "#")
    link_info = f"For more details, visit the event website at {link}."
    # Combining all the parts
    paragraph = f"{name} is scheduled to take place {date_range}. {description} {link_info}"
    paragraph = paragraph.replace("\n", "")
    return "```"+paragraph+"```"

class EventInfoToolSpec(BaseToolSpec):
    """Event information tool spec."""
    spec_functions = ["google_search"]

    def __init__(self, key: str, engine: str) -> None:
        """Initialize with parameters."""
        self.key = key
        self.engine = engine

    def formatted_event_info(self, query: str):
        """
        Make a query to the Google search engine to receive formatted event info.
        Each event is delimited by triple backticks.

        Args:
            query (str): The query to be passed to Google search. This should include dates when users are most free.
        """
        base_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.key,
            "cx": self.engine,
            "q": query
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            search_results = response.json()

            # Extract and parse the relevant information from the response
            parsed_results = []
            if "items" in search_results:
                for item in search_results["items"]:
                    parsed_result = {
                        "title": item.get("title"),
                        "link": item.get("link"),
                        "snippet": item.get("snippet"),
                        "rich_snippets": item.get("pagemap", {})
                    }
                    parsed_results.append(parsed_result)
            # print(parsed_results)

            event_info = ''
            for result in parsed_results:
                if 'rich_snippets' in result:
                    processed_snippet = process_rich_snippets(result['rich_snippets'])
                    # print(processed_snippet)
                    if processed_snippet != []:
                        for event in processed_snippet:
                            event_info += event_dict_to_paragraph(event)
                            event_info += '\n\n'

            return event_info

        except requests.exceptions.RequestException as e:
            print(f"Error making the request: {e}")
            return ''

class GoogleSearchToolSpec(BaseToolSpec):
    """Google Search tool spec."""
    spec_functions = ["google_search"]

    def __init__(self, key: str, engine: str, num: Optional[int]=10) -> None:
        """Initialize with parameters."""
        self.key = key
        self.engine = engine
        self.num = num

    def google_search(self, query: str):
        """
        Use when users ask for more details about an event or activity.

        Args:
            query (str): The query to be passed to Google search.
            num (int, optional): The number of search results to return. Defaults to 10.

        Raises:
            ValueError: If the 'num' is not an integer between 1 and 10.
        """
        QUERY_URL_TMPL = ("https://www.googleapis.com/customsearch/v1?key={key}&cx={engine}&q={query}")
        url = QUERY_URL_TMPL.format(
            key=self.key, engine=self.engine, query=urllib.parse.quote_plus(query)
        )

        if self.num is not None:
            if not 1 <= self.num <= 10:
                raise ValueError("num should be an integer between 1 and 10, inclusive")
            url += f"&num={self.num}"

        response = requests.get(url)
        return [Document(text=response.text)]