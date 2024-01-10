import os
import json
import boto3
import requests
import openai
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_model.dialog import ElicitSlotDirective
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.utils import is_request_type, is_intent_name
from ask_sdk_model import Response
from token_handler import handle_tokens


ssm = boto3.client('ssm')

class DailySummaryIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("DailySummaryIntent")(handler_input)

    def handle(self, handler_input):
        FITBIT_ACCESS_TOKEN = handle_tokens()
        OPENAI_API_KEY      = get_parameter('OPENAI_API_KEY')

        # Query Fitbit API
        headers = {'Authorization': 'Bearer ' + FITBIT_ACCESS_TOKEN}
        sleep_data = requests.get('https://api.fitbit.com/1/user/-/sleep/date/today.json', headers=headers).json()
        food_data = requests.get('https://api.fitbit.com/1/user/-/foods/log/date/today.json', headers=headers).json()
        exercise_data = requests.get('https://api.fitbit.com/1/user/-/activities/date/today.json', headers=headers).json()

        # Get age and weight from Fitbit
        age = int(sleep_data['summary']['totalMinutesAsleep']) 
        weight = float(exercise_data['summary']['caloriesOut'])  

        food_items = []
        for entry in food_data['foods']:
            food_items.append(f"{entry['name']} ({entry['calories']} calories)")
        food_list = ", ".join(food_items)
        prompt = f"For someone of age {age} and weight {weight}, I ate {food_list} today with a goal of being healthy, at a good body weight percentage and strong, how was my day? Tell me if there are any improvements I can make if I am failing in an area."

        # Call OpenAI API
        openai.api_key = OPENAI_API_KEY
        openai_response = openai.Completion.create(
            engine="gpt-4-1106-preview",
            prompt=prompt,
            max_tokens=60
        )

        # Return OpenAI response
        speech_text = openai_response.choices[0].text
        handler_input.response_builder.speak(speech_text).set_should_end_session(True)
        return handler_input.response_builder.response

class LaunchRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        speak_output = "Welcome to NickLife, the hub for all your Nick-related needs!"
        return handler_input.response_builder.speak(speak_output).ask(speak_output).response

class StopIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("AMAZON.StopIntent")(handler_input)

    def handle(self, handler_input):
        speak_output = "Thanks for talking, I'll be here for all your Nick-related inquiries"
        return handler_input.response_builder.speak(speak_output).set_should_end_session(True).response

class CancelIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("AMAZON.CancelIntent")(handler_input)

    def handle(self, handler_input):
        speak_output ="Thanks for talking, I'll be here for all your Nick-related inquiries"
        return handler_input.response_builder.speak(speak_output).set_should_end_session(True).response


sb = SkillBuilder()
sb.add_request_handler(DailySummaryIntentHandler())
sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(StopIntentHandler())
sb.add_request_handler(CancelIntentHandler())

lambda_handler = sb.lambda_handler()



