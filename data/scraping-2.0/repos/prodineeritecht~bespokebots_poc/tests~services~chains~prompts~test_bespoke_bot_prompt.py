import pytest
from typing import List, Optional

from langchain.schema import AIMessage, HumanMessage, SystemMessage

from bespokebots.services.chains.prompts import (
    BESPOKE_BOT_MAIN_TEMPLATE,
    OUTPUTPARSER_TEMPLATE
    )
from bespokebots.services.chains.prompts.bespoke_bot_prompt import BespokeBotChatPrompt

def test_bespoke_bot_prompt():
    """Test the BespokeBotChatPrompt class."""
    prompt = BespokeBotChatPrompt.from_user_request("Hello")
    assert prompt is not None

    messages = prompt.format_prompt(
        assistant_name="BespokeBot",
        client_request="Hello",
        tools=[],
        response_format=OUTPUTPARSER_TEMPLATE,
    ).to_messages()

    assert len(messages) == 2
    assert any(
        isinstance(msg, HumanMessage) for msg in messages
    ), "No HumanMessage in the list"
    assert any(
        isinstance(msg, SystemMessage) for msg in messages
    ), "No SystemMessage in the list"

def test_bespoke_bot_prompt_with_tools():
    """Test the BespokeBotChatPrompt class."""
    prompt = BespokeBotChatPrompt.from_user_request("Hello")
    assert prompt is not None

    tools_list = (
        f"""1. view_calendar_events""",
        f"""2. view_calendar_events_by_date""",
        f"""3. create_calendar_event""",
        f"""4. delete_calendar_event""",
        f"""5. update_calendar_event""",
        f"""6. analyze_calendar_events""",
        f"""7. save_to_memory""",
        f"""8. finish"""
    )

    tools_text = "\n".join(tools_list)

    formatted_prompt = prompt.format_prompt(
        assistant_name="BespokeBot",
        tools=tools_text,
        response_format=OUTPUTPARSER_TEMPLATE,
    )

    messages = formatted_prompt.to_messages()
    assert len(messages) == 2
    assert any(
        isinstance(msg, HumanMessage) for msg in messages
    ), "No HumanMessage in the list"
    assert any(
        isinstance(msg, SystemMessage) for msg in messages
    ), "No SystemMessage in the list"

    prompt_text = formatted_prompt.to_string()

    assert (
        str.find(prompt_text, tools_text) != -1
    )
