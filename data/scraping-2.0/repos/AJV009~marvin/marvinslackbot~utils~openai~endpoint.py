import openai
import json
import os
from os.path import exists
import random
from marvinslackbot.utils.slack.helpers import SlackHelpers
from marvinslackbot.utils.autogpt.autogpt import AutoGPT


class OpenAIHelpers:
    def __init__(self):
        openai.organization = os.environ.get("OPENAI_ORG_ID")
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        self.thiking_file = "marvinslackbot/data/thinking.txt"
        self.thinking_thoughts = []
        self.slack_helpers = SlackHelpers(init_self_app=True)
        self.messages = []
        with open("marvinslackbot/data/prompts.json", "r") as f:
            self.prompts = json.load(f)
        with open("marvinslackbot/data/flags.json", "r") as f:
            self.flags = json.load(f)

    # Prefetch thinking thoughts
    def thinking_setup(self):
        if (exists(self.thiking_file)):
            with open(self.thiking_file, "r") as f:
                lines = f.readlines()
                if len(lines) > 0:
                    self.thinking_thoughts = lines
        if (len(self.thinking_thoughts) <= 0):
            self.thinking_thoughts = ["I'm thinking..."]

    # Returns a random thinking thought
    def thinking(self):
        if (len(self.thinking_thoughts) > 0):
            return self.flags["init_thinking"] + random.choice(self.thinking_thoughts)

    # Sets the data for the chat
    def set_data(self, messages):
        self.messages = messages
        if (len(messages) > 0):
            return {"data_type": "thread", "moderation": False}
        else:
            return {"data_type": "single", "moderation": self.moderate(messages["message"])}

    # Moderates the text
    def moderate(self, text):
        moderation_response = self.openai.Moderation.create(input=text)
        if moderation_response["results"][0]["flagged"]:
            return self.flags["warning"]
        return False

    # Prepares the data for the chat API
    def data_prep(self):
        openai_messages = [
            {"role": "system", "content": self.prompts["system"]}]
        if (len(self.messages) > 0):
            bot_id = self.slack_helpers.get_bot_id()

            for message in self.messages:
                # Check for bot messages
                if message["user_id"] == bot_id:
                    # Avoid messages with flags
                    if not message["message"].startswith(tuple(self.flags.values())):
                        openai_messages.append({
                            "role": "assistant",
                            "content": message["message"]
                        })
                else:
                    # Add user messages
                    openai_messages.append({
                        "role": "user",
                        "content": f'[{message["user"]}] {message["message"]}'
                    })

            self.messages = openai_messages

    def model_extractor(self):
        MODELS = {"[MARVIN-GPT4]": "gpt-4", "[MARVIN-GPT3]": "gpt-3.5-turbo", "[MARVIN-AUTOGPT]": "autogpt"}
        DEFAULT_MODEL = "gpt-3.5-turbo"

        model = DEFAULT_MODEL
        model_found = False

        for message in reversed(self.messages):
            new_content = message["content"]
            for identifier, model_name in MODELS.items():
                if identifier in new_content:
                    if not model_found:
                        model = model_name
                        model_found = True
                    new_content = new_content.replace(identifier, "")
            message["content"] = new_content

        return model

    def quickChat(self, messages, model):
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            )
        return response['choices'][0]['message']['content']
    
    def return_openai(self):
        return openai

    def chat(self):
        # empty response
        response = None

        # Prepare data for the chat request
        self.data_prep()

        # Extract the model from messages and update messages
        model = self.model_extractor()

        # Custom AutoGPT Model
        if model == "autogpt":
            auto_gpt = AutoGPT(self.messages)
            self.messages = auto_gpt.get_messages()
            model = auto_gpt.get_model()

        # simple responses
        if response is None:
            # Send the chat request to OpenAI
            response = openai.ChatCompletion.create(
                model=model,
                messages=self.messages,
            )

        # Return the content of the response
        return response['choices'][0]['message']['content']
