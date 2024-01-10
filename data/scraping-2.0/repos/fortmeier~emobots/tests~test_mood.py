from openai import OpenAI
import emobots.mood as mood

import logging

logging.basicConfig(
    filename="generate_random_person_output.log",
    encoding="utf-8",
    level=logging.DEBUG,
    force=True,
)


def test_mood_analysis():
    client = OpenAI()

    name = "Harold Wetherby"
    chat_messages = "I'm feeling great today!"

    mood_analyis_system_prompt = f"""
        From the above conversation, what do you think is the current mood of {name}?
        If uncertain, simply state that the mood is unknown."""

    mood_analysis_response = mood.mood_analysis(
        client, name, chat_messages, mood_analyis_system_prompt
    )

    assert mood_analysis_response != None
    assert len(mood_analysis_response) > 0
