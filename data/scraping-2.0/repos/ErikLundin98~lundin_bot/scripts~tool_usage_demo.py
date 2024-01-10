from box import Box
from dotenv import load_dotenv
import yaml
from assistant.constants import OPENAI_API_KEY, OPENAI_ORGANIZATION
from assistant.language_model.action import run_action
from assistant.language_model.model import LanguageModel
from assistant.language_model.tools import select_action

import os

load_dotenv()

with open("config.yaml", "r") as file:
    config = Box(yaml.safe_load(file))

llm = LanguageModel(config)

query = "What is the hue and saturation of the color red?"

action, message = select_action.main(
    query=query,
    llm=llm,
    config=config
)
print(message)

response = run_action(
    action=action,
    query=query,
    llm=llm,
    config=config,
)
print(response)

