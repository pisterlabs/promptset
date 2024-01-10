import pytest
from typing import List, Optional

from langchain.schema import AIMessage, HumanMessage, SystemMessage

from bespokebots.services.chains.prompts import CalendarAnalysisChatPrompt
from bespokebots.services.chains.calendar_templates import (
    CalendarDataAnalyzerTemplates as Templates,
)


def events() -> dict:
    return {
        "events": [
            {
                "event_id": "6tukcd7qohtrijv8kic8limlvk_20230602",
                "title": "Kids with me",
                "start": "2023-05-29",
                "end": "2023-06-08",
            },
            {
                "event_id": "_88qkacho6ks3eba569238b9k8cpj8b9o6opk4ba388s4aga16sq3gghl6o",
                "title": "Kel in South Dakota",
                "start": "2023-05-30",
                "end": "2023-06-06",
            },
            {
                "event_id": "hsj31m6lrjd617m5b8meam59e8",
                "start": "2023-05-31T10:00:00-04:00",
                "end": "2023-05-31T11:00:00-04:00",
                "description": "Appointment added to your calendar by Tom, your AI Assistant",
                "summary": "Therapy w/ Alex",
            },
            {
                "event_id": "5o7oodp4s58umohm8p6tdkb4i0",
                "start": "2023-05-31T13:45:00-04:00",
                "end": "2023-05-31T15:00:00-04:00",
                "description": "Appointment added to your calendar by Tom, your AI Assistant",
                "summary": "Physical exam w/ Dr. Curran",
            },
            {
                "event_id": "55cuapjjkaqdp10a01obt88p2s",
                "start": "2023-05-31T16:00:00-04:00",
                "end": "2023-05-31T17:00:00-04:00",
                "summary": "Absolute Chiro",
            },
            {
                "event_id": "39tfjkkqoavueihtllsi6g8nto",
                "start": "2023-05-31T17:00:00-04:00",
                "end": "2023-05-31T17:30:00-04:00",
                "description": "Pick up kids from school",
                "summary": "Pick up kids",
            },
            {
                "event_id": "ru89l98hleiq9gnppi3nl9eqd8",
                "start": "2023-06-01T13:00:00-04:00",
                "end": "2023-06-01T14:00:00-04:00",
                "description": "Appointment with Dr. Hayner at SeaCoast Hand Therapy",
                "summary": "SeaCoast Hand Therapy",
                "location": "Scarborough, ME",
            },
            {
                "event_id": "73ln88bvum4nstil015ql3ljkc",
                "start": "2023-06-01T17:00:00-04:00",
                "end": "2023-06-01T17:30:00-04:00",
                "description": "Pick up kids from school",
                "summary": "Pick up kids",
                "location": "School",
            },
        ]
    }


def test_chat_prompt_with_user_requirements():
    user_requirements = [
        "- Ensure there is no less than 30 minutes of free time before and after any event that needs to be scheduled.",
        "- On any given day, there should be as few appointments as possible, so when choosing between two or more days on which to schedule an appointment, choose the day with the least number of appointments.",
    ]
    separator = "\n\t"
    user_requirements_template = Templates.USER_REQUIREMENTS_TEMPLATE(
        separator.join(user_requirements)   
    )
    human_question = """I want to go to a Vinyasa yoga class either on Thursday 5/31/2023 or Friday 6/1/2023, both at 9:30 AM EST for one hour. Today is Tuesday 5/30/2023. Could you please check my calendar and tell me on which day, 5/31 or 6/1, I don’t have any existing events that the yoga class would overlap?"""
    prompt = CalendarAnalysisChatPrompt.from_user_request(human_question)
    assert prompt is not None
    
    messages = prompt.format_prompt(
        events_to_analyze=events(),
        user_requirements=user_requirements_template,
        output_parser_template=Templates.OUTPUTPARSER_TEMPLATE,
    ).to_messages()

    assert len(messages) == 2
    assert any(
        isinstance(msg, HumanMessage) for msg in messages
    ), "No HumanMessage in the list"
    assert any(
        isinstance(msg, SystemMessage) for msg in messages
    ), "No SystemMessage in the list"

    formatted_prompt = prompt.format_prompt(
        events_to_analyze=events(),
        user_requirements=user_requirements_template,
        output_parser_template=Templates.OUTPUTPARSER_TEMPLATE,
    ).to_string()
    assert formatted_prompt is not None
    assert (
        str.find(
            formatted_prompt,
            "Ensure there is no less than 30 minutes of free time before and after any event that needs to be scheduled",
        )
        > -1
    )

def test_chat_prompt_without_user_requirements():
    human_question = """I want to go to a Vinyasa yoga class either on Thursday 5/31/2023 or Friday 6/1/2023, both at 9:30 AM EST for one hour. Today is Tuesday 5/30/2023. Could you please check my calendar and tell me on which day, 5/31 or 6/1, I don’t have any existing events that the yoga class would overlap?"""
    prompt = CalendarAnalysisChatPrompt.from_user_request(human_question)
    assert prompt is not None

    formatted_prompt = prompt.format_prompt(
        events_to_analyze=events(),
        user_requirements="",
        output_parser_template=Templates.OUTPUTPARSER_TEMPLATE,
    ).to_string()
    assert formatted_prompt is not None
    assert (
        str.find(
            formatted_prompt,
            "Here are the user's additional requirements:",
        )
        == -1
    )
