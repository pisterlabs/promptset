# Importing necessary libraries
import json
import os
from datetime import datetime

import openai
import toml

# Setting the environment variable for OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = "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"


class Agent:
    def __init__(self, config_path: str = ""):
        # System prompts
        self.system: str = ""
        self.instructions: str = ""

        self.concerned: bool = False
        self.templates: dict = {}

        self.knowledge: list = []
        self.messages: list = []

        if config_path:
            self.load_config(config_path)

    def load_config(self, path):
        # Loading the agent from the config file
        with open(path) as f:
            agent = toml.load(f)

        # Extracting core and instructions from the agent
        self.system = agent["system"]
        self.instructions = agent["instructions"]
        self.request = agent["request"]

        # Extracting templates
        self.templates["knowledge"] = agent["knowledge"]

    def think(self, statement):
        self.messages.append({"role": "user", "content": statement})
        self.messages = self.messages[-6:]

        messages_start = [
            {"role": "user", "content": self.system},
            {
                "role": "user",
                "content": "Here is our most recent conversation: \n",
            },
        ]

        messages_end = [
            # {"role": "user", "content": self.instructions},
            {"role": "user", "content": self.request},
        ]

        messages = messages_start + self.messages + messages_end

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        
        self.messages.append(
            {
                "role": "assistant",
                "content": json.loads(response.choices[0].message.content.strip())['response'], # type: ignore
            }
        )
        return response


def log(s):
    with open("therapist_turbo.md", "a") as log:
        log.write("# " + str(s) + "\n")


log("---\n# NEW SESSION: {}\n".format(datetime.now().strftime("%B %d, %Y %H:%M:%S")))
therapist = Agent(config_path="./agents/therapist.toml")
message = "Hi"
while True:
    therapist.load_config("./agents/therapist.toml")
    response = therapist.think(message)
    response = json.loads(response.choices[0].message.content.strip())['response'] # type: ignore
    print(response)
    
    log("---\n# Notes: {}".format(input(">>> Bugs or comments?: ")))
    message = input("> ")
