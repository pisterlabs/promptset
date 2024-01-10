# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet, FollowupAction
from rasa_sdk.executor import CollectingDispatcher
import random
import os
import sys
import openai

# Add "/app/actions" to the sys.path
actions_path = os.path.abspath("/app/actions")
sys.path.insert(0, actions_path)

print("-#-System-path-#-")
for path in sys.path:
    print(path)
print("-#-END-OF-System-path-#-")
# Import search_content.py from /actions folder
from search_content import main_search


# Import api key from secrets
secret_value_0 = os.environ.get("openai")

openai.api_key = secret_value_0
# Provide your OpenAI API key

def generate_openai_response(query, model_engine="text-davinci-002", max_tokens=124, temperature=0.8):
    """Generate a response using the OpenAI API."""
    
    # Run the main function from search_content.py and store the results in a variable
    results = main_search(query)

    # Create context from the results
    context = "".join([f"#{str(i)}" for i in results])[:2014] # Trim the context to 2014 characters - Modify as necessory
    prompt_template = f"Relevant context: {context}\n\n Answer the question in detail: {query}"

    # Generate a response using the OpenAI API
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt_template,
        max_tokens=max_tokens,
        temperature=temperature,
        n=1,
        stop=None,
    )

    return response.choices[0].text.strip()

class GetOpenAIResponse(Action):

    def name(self) -> Text:
        return "action_get_response_openai"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Use OpenAI API to generate a response
        query = tracker.latest_message.get('text')
        response = generate_openai_response(query)
                
        # Output the generated response to user
        dispatcher.utter_message(text=response)
                
class GeneralHelp(Action):
    def name(self) -> Text:
        return "action_general_help"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        user_role = tracker.slots.get("user_role", None)
        
        if user_role is None:
            dispatcher.utter_message(text="Sure! Are you a developer or a client representing an organization?")
        else:
            return [FollowupAction("action_help_with_role")]

# Modified from @Rohit Garg's code https://github.com/rohitkg83/Omdena/blob/master/actions/actions.py
class ActionHelpWithRole(Action):

    def name(self) -> Text:
        return "action_help_with_role"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Get the value of the first_occurrence_user_type slot
        current_user_type = tracker.slots.get("user_role", None)
   
        if current_user_type == 'developer':
            msg = "Thanks a lot for providing the details. You can join one of our local chapter and collaborate on " \
                    "various projects and challenges to Develop Your Skills, Get Recognized, and Make an Impact. Please " \
                    "visit https://omdena.com/community for more details. Do you have any other questions? "

        elif current_user_type == 'client':
            msg = "Thanks a lot for providing the details. With us you can Innovate, Deploy and Scale " \
                    "AI Solutions in Record Time. For more details please visit https://omdena.com/offerings. Do you have any other questions? "
        else:
            msg = "Please enter either developer or client"

        dispatcher.utter_message(text=msg)

class ResetSlotsAction(Action):
    def name(self) -> Text:
        return "action_reset_slots"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        slots_to_reset = ["user_role"]  # Add the names of the slots you want to reset
        events = [SlotSet(slot, None) for slot in slots_to_reset]
        return events

class ActionJoinClassify(Action):

    def name(self) -> Text:
        return "action_join_classify"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Get the value of the latest intent 
        last_intent = tracker.slots.get("local_chapter", None)

        # Check if the last intent was 'local_chapter'
        if last_intent == 'local chapter':
            dispatcher.utter_message(template="utter_join_chapter")
        else:
            return [FollowupAction("action_get_response_openai")]
            


class ActionEligibilityClassify(Action):

    def name(self) -> Text:
        return "action_eligibility_classify"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Get the value of the latest intent 
        last_intent = tracker.slots.get("local_chapter", None)

        # Check if the last intent was 'local_chapter'
        if last_intent == 'local chapter':
            dispatcher.utter_message(template="utter_local_chapter_participation_eligibility")
        else:
            return [FollowupAction("action_get_response_openai")]

 
class ActionCostClassify(Action):

    def name(self) -> Text:
        return "action_cost_classify"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Get the value of the latest intent 
        last_intent = tracker.slots.get("local_chapter", None)

        # Check if the last intent was 'local_chapter'
        if last_intent == 'local chapter':
            dispatcher.utter_message(template="utter_local_chapter_cost")
        else:
            return [FollowupAction("action_get_response_openai")]

class SayHelloWorld(Action):

    def name(self) -> Text:
        return "action_hello_world"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Use OpenAI API to generate a response
        secret_value_0 = os.environ.get("openai")
        openai.api_key = secret_value_0
        model_engine = "text-davinci-002"
        prompt_template = "Say hello world"

        response = openai.Completion.create(
            engine=model_engine,
            prompt=prompt_template,
            max_tokens=124,
            temperature=0.8,
            n=1,
            stop=None,
        )

        # Output the generated response to user
        generated_text = response.choices[0].text
        dispatcher.utter_message(text=generated_text)