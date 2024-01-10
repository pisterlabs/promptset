import io

import json

import pytest
from openai.types.chat import ChatCompletionMessage

from functioncalming import get_completion, get_client
from tests.conftest import MockOpenAI


def get_weather(city: str, zip: str | None = None) -> str:
    """
    Get the weather
    :param city: city name
    :param zip: zip code
    :return: the weather
    """
    return "lots of snow"


def get_time(city: str, zip_code: str | None = None) -> str:
    return "7pm"


@pytest.mark.asyncio
async def test_simple_function_call():
    (weather_results, *_), message_history = await get_completion(
        user_message="What's the weather like in Berlin?",
        tools=[get_weather, get_time],
        pass_results_to_model=True
    )
    assert "snow" in json.dumps(message_history)
    assert weather_results == "lots of snow"


@pytest.mark.asyncio
async def test_wrong_function_name():
    mock_client = MockOpenAI(get_client())
    mock_client.add_next_responses(
        ChatCompletionMessage(**{
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {
                        "name": "does_not_exist",
                        "arguments": "{}"
                    }
                },
            ]
        }),
        # now comes a failure response, then:
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "id": "3",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Munich"}'
                    }
                },
            ]
        },
    )
    with io.StringIO() as fake_file:
        history = []
        (weather_results, *_), message_history = await get_completion(
            history=history,
            user_message="What's the weather like in Berlin?",
            tools=[get_weather, get_time],
            pass_results_to_model=True,
            rewrite_history_in_place=False,
            rewrite_log_destination=fake_file,
            retries=2,
            openai_client=mock_client
        )
        file_content = fake_file.getvalue()
    assert "does not exist" in str(history)
    assert "snow" in weather_results

# TODO simulate the model calling a function incorrectly and make sure a retry is done

# TODO make sure that retry information is hidden in the clean message history

# TODO simulate a correct function call that, internally, raises a validation error,
#  and make sure this does not lead to a retry!
