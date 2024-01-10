import os
import time
import logging
import openai
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.environ.get('OPENAI_API_KEY')


def input_parser(user_request : str, selectedCityID : str, user_id : str) -> str:
    """
    This function takes the user request and the selected city ID and user ID and returns the query in a more structured and concise format.

    Args:
        user_request (str): The user request.
        selectedCityID (str): The selected city ID.
        user_id (str): The user ID.
    
    Example:
        >>> input_parser("I want to go to barcelona for the weekend on 12th of january. Outbound flight departure after 4pm. Direct flights.", "madrid_es", "TestUser")
        Origin: Madrid, ES | Destination: Barcelona, ES | Departure: 12.1. after 4pm | Duration: Weekend | Flights: Direct
    
    Returns:
        str: The parsed input.
    """

    start_time = time.time()
    logger.debug("[UserID: %s] Parsing user_request", user_id)

    # Create the prompt templates
    system_template = """INSTRUCTIONS:
You're an intelligent AI agent. You are going to get user's description about a flight they are looking for. Your job is to formulate user requests in a structured and concise manner, so that another trained AI flight search system can handle the request more easily.
Example ot the desired output: "Origin: Stockholm, SE | Destination: Somewhere in Eastern Europe | Departure: March 2024 | Duration: Weekend | Flights: max 1 layover"
"""

    #example 1
    userExample1 = """Origin: madrid_es
User request: I want to go to barcelona for the weekend on 12th of january. Outbound flight departure after 4pm. Direct flights."""

    botExample1 = """Origin: Madrid, ES | Destination: Barcelona, ES | Departure: 12.1. after 4pm | Duration: Weekend | Flights: Direct"""

    #example 2
    userExample2 = """Origin: munich_de
User request: Two-week trip to somewhere in South America. Departure in January."""

    botExample2 = """Origin: Munich, DE | Destination: South America | Departure: January | Duration: 2 weeks"""

    human_template = f"Origin: {selectedCityID}\nUser request: {user_request}"

  # Construct the conversation message list
    message_list = [
        {"role": "system", "content": system_template},
        {"role": "user", "content": userExample1},
        {"role": "assistant", "content": botExample1},
        {"role": "user", "content": userExample2},
        {"role": "assistant", "content": botExample2},
        {"role": "user", "content": human_template}
    ]

    # Request the response from the model
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-0613",
      temperature=0,
      messages=message_list,
    )
    response_content = response.choices[0].message['content']

    logger.debug("[UserID: %s] Parsed input: %s", user_id, response_content)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.debug("[UserID: %s] Function execution time: %s seconds", user_id, elapsed_time)

    return response_content
