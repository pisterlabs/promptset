from openai import OpenAI
from emobots.random_person import create_random_person_description

import logging

logging.basicConfig(
    filename="generate_random_person_output.log",
    encoding="utf-8",
    level=logging.DEBUG,
    force=True,
)


def test_generate_random_person():
    client = OpenAI()

    description = create_random_person_description(client)
    assert description != None
    assert len(description) > 0

    logging.debug(description)
