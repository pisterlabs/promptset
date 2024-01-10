from openai import OpenAI
import emobots.intention as intention

import logging

logging.basicConfig(
    filename="generate_random_person_output.log",
    encoding="utf-8",
    level=logging.DEBUG,
    force=True,
)


def test_intention_analysis():
    client = OpenAI()

    name = "Harold Wetherby"
    chat_messages = "I'm feeling great today!"

    intention_analyis_system_prompt = f"""
        From the above conversation, what do you think is the current intention of {name}?
        If uncertain, simply state that the intention is unknown."""

    intention_analysis_response = intention.intention_analysis(
        client, name, chat_messages, intention_analyis_system_prompt
    )

    assert intention_analysis_response != None
    assert len(intention_analysis_response) > 0
