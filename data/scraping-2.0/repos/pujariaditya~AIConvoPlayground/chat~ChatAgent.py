import openai

import logging

from model_api.ModelClient import ModelClient
from model_api.model_urls import model_urls, gptmodels
logger = logging.getLogger(__name__)

class ChatAgent:
    def __init__(self, char_name, char_persona,char_greeting,world_scenario,example_dialogue, model_name):
        self.char_name  = char_name
        self.char_persona = char_persona
        self.char_greeting = char_greeting
        self.world_scenario = world_scenario
        self.example_dialogue = example_dialogue
        self.model_name = model_name
        self.is_running = True

    def build_prompt_for(self,chat_history,user_message):
        str_chat_history = [' '.join(t) for t in chat_history]
        example_history = str_chat_history if self.example_dialogue else []

        concatenated_history = [*example_history, *str_chat_history]

        # Construct the base turns with the info we already have.
        prompt_turns = [
            "<START>",
            *concatenated_history[-8:],
            f"You: {user_message}",
            f"{self.char_name}:",
        ]

        # If we have a scenario or the character has a persona definition, add those
        # to the beginning of the prompt.
        if self.world_scenario:
            prompt_turns.insert(
                0,
                f"Scenario: {self.world_scenario}",
            )

        if self.char_persona:
            prompt_turns.insert(
                0,
                f"{self.char_name}'s Persona: {self.char_persona}",
            )

        prompt_str = "\n".join(prompt_turns)
        return prompt_str

    def generate_message(self, chat_history,user_message):

        payload = {
            "message": self.build_prompt_for(chat_history,user_message)
        }
        if self.model_name in gptmodels:
            logger.error("GPT")
            response = openai.ChatCompletion.create(model=self.model_name, messages = [
            {"role": "user", "content": payload["message"]}
        ])
            return response['choices'][0]['message']['content']
        elif self.model_name in list(model_urls.keys()):
            logger.error("CUSTOM MODEL")
            server_address = model_urls[self.model_name]
            client = ModelClient(server_address)
            response = client.generate_text( payload["message"])
            return response if response else "Internal request failed"
        else:
            return "Internal request failed"

    def stop(self):
        self.is_running = False
