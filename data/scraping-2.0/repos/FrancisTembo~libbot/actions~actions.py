import logging
import os

from rasa_sdk import Action

import openai

from .llm import LanguageModel

# Configure logging
logging.basicConfig(filename="rasa_action.log", level=logging.ERROR)

openai
language_model = LanguageModel(
    openai_key="",
    model="gpt-3.5-turbo-16k",
    temperature=0.7,
)


class ActionAskOpenAI(Action):
    def name(self) -> str:
        return "action_ask_openai"

    def run(self, dispatcher, tracker, domain):
        try:
            # Get the latest message from the user
            message = tracker.latest_message["text"]

            # Create a conversation with the OpenAI API
            completion = language_model.generate(human_input=message)

            if completion:
                dispatcher.utter_message(text=completion)
            else:
                # Send an error message if no completion
                dispatcher.utter_message(text="I'm sorry, I can't assist with that.")
        except Exception as e:
            # Log the error
            logging.error(f"Error in Rasa action: {str(e)}")
            # Send an error message to the user
            dispatcher.utter_message(
                text="An error occurred while processing your request. Please try again later."
            )

        return []
