import asyncio
import openai
import json
import os
import datetime
import pytz
import aiofiles

from operator import itemgetter
from event_search import survey_defaults, randomize

# Set up your OpenAI API credentials
openai.api_key = os.getenv("OPEN_AI_KEY")

async def call_chat_gpt(messages):
    loop = asyncio.get_event_loop()

    response = await loop.run_in_executor(None, lambda: openai.ChatCompletion.create(
        model = "gpt-4-0613",
        messages = messages
    ))

    return response["choices"][0]["message"]

async def generate_random_event():
    values = survey_defaults()

    randomed = randomize(values)
    randomed['location'] = "South Lake Union"

    return await generate_event_sponsorless(randomed)


async def generate_sponsor(sponsored_location, sponsored_activity, itinerary):
    example_json = {
        "activity_name": "Sponsored Activity",
        "activity_location": "Sponsored Location",
        "distance_from_previous": 0.5,
        "time_elapsed_minutes": 60,
        "cost_per_person": 30,
        "description": "Participate in the Sponsored Activity at the Sponsored Location.",
        "sponsored": True
    }
    example_json = json.dumps(example_json)
    example_response = {
        "activity_name": "Monopoly Mania",
        "activity_location": "Cherry Street Coffee, 2719 1st Ave, Seattle, WA 98121",
        "distance_from_previous": 0.5,
        "time_elapsed_minutes": 60,
        "cost_per_person": 30,
        "description": "Engage in a thrilling game of Monopoly at Cherry Street Coffee. Enjoy the aroma of freshly brewed coffee as you strategize your way to victory, all while immersing yourself in the cozy ambiance of this beloved Seattle coffee shop.",
        "sponsored": True
    }
    example_response = json.dumps(example_response)
    request_prompt = f"Fill the following .json travel itinerary so that the Sponsored Location is {sponsored_location} and sponsored activity is {sponsored_activity}. Change activity_name, description (be creative about the description); don't change anything else {itinerary}"

    messages = [
        {"role": "system", "content": "You are a helpful assistant that fills out itineraries for events based on provided details. YOU ONLY RESPOND WITH JSON. DON'T INCLUDE ANY MESSAGES ONLY INCLUDE FORMATTED JSON. JSON MUST USE DOUBLE QUOTES"},
        {"role": "user", "content": f"Fill the following .json travel itinerary so that the Sponsored Location is Cherry Street Coffee, 2719 1st Ave, Seattle, WA 98121 and sponsored activity is playing Monopoly. Change activity_name, description (be creative about the description); don't change anything else {example_json}"},
        {"role": "system", "content": f"{example_response}"},
        {"role": "user", "content": request_prompt}
    ]

    while True:
        try:
            response = (await call_chat_gpt(messages)).to_dict()
            json_response = json.loads(response['content'])
            break
        except Exception as e:
            print("An error occured ", e)

    return json_response

async def generate_event_sponsorless(parameter_dict):
    distance, location, environment, transportation, activities, budget, duration = itemgetter('distance', 'location', 'environment', 'transportation', 'activities', 'budget', 'duration')(parameter_dict)

    template_prompt_sponsorless = "Generate an itinerary  for an event for 6 people within {} miles of {} that is mostly {}. The group with travel via {}. The group would like to {}. The budget of each person in the group is {} dollars. The event lasts for {} hours; use time elapsed for each step not start/endtime (give it in integer minutes). Include an activity in the itinerary that sends the group to *Sponsored Location* to do *Sponsored activity*(leave those entries blank, add fields to all entries sponsored true/false)"

    now = datetime.datetime.now().astimezone(pytz.timezone('US/Pacific')).strftime("%H:%M")

    prompt = template_prompt_sponsorless.format(distance, location, environment, transportation, activities, budget, now, duration)

    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_directory, "sample_response_sponsorless.json")

    async with aiofiles.open(file_path, mode='r') as file:
                content = await file.read()
                print("Updated data")
                json_data = json.loads(content)

    sample_response = json.dumps(json_data)

    messages = [
        {"role": "system", "content": " You are a helpful assistant that writes itineraries for events based on provided details. YOU ONLY RESPOND WITH JSON."},
        {"role": "user", "content": "Generate an itinerary  for an event for 6 people within 2 miles of South Lake Union in Seattle, WA that is mostly Outdoors. The group with travel via Rental Scooters/Bikes. The group would like to Explore the city. The budget of each person in the group is 120 dollars. The event lasts for 4 hours; use time elapsed for each step not start/endtime (give it in integer minutes). Include an activity in the itinerary that sends the group to Cherry Street Coffee (2719 1st Ave, Seattle, WA 98121) to do playing Monopoly"},
        {"role": "assistant", "content": sample_response},
        {"role": "user", "content": prompt},
    ]
    #Retries GPT calls until it works
    while True:
        try:
            response = (await call_chat_gpt(messages)).to_dict()
            json_response = json.loads(response['content'])
            break
        except Exception as e:
            print("An error occured ", e)

    survey_dict = {
        "distance": distance,
        "location": location,
        "environment": environment,
        "transportation": transportation,
        "activities": activities,
        "budget": budget,
        "duration": duration,
        "sponsored_location": None,
        "sponsored_activity": None,
        "itinerary": json_response
    }
    return survey_dict

async def fill_sponsor(sponsored_location, sponsored_activity, data):
    # Find the sponsored activity
    for i, activity in enumerate(data["itinerary"]["activities"]):
        if activity["sponsored"]:
            # Call the function to generate sponsor and replace the old activity
            data["itinerary"]["activities"][i] = await generate_sponsor(sponsored_location, sponsored_activity, activity)
    return data
