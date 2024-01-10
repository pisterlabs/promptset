from openai import OpenAI

import logging

logging.basicConfig(
    filename="random_person.log", encoding="utf-8", level=logging.INFO, force=True
)


from api_key import api_key
from emobots.emobot import Emobot

from emobots.random_person import *
from emobots.personas import get_persona

from emobots.tools import get_name_from_description

from emobots.chat import console_interaction_loop_instant

client = OpenAI(api_key=api_key)

# person_desc = create_random_person_description(client)
# name = get_name_from_description(client, person_desc)

name, person_desc, *others = get_persona("deamon_malgorth")

logging.info(person_desc)

bot = Emobot(client, name, person_desc)

console_interaction_loop_instant(bot, max_line_length=80)
