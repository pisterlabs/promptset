import json

import openai
import pytest
from langchain.prompts import ChatPromptTemplate
from langchain.schema.messages import AIMessage, HumanMessage

from llm_api.backends.openai import OpenaiCaller, OpenaiModelCallError

pytest_plugins = ("pytest_asyncio",)


def test_openai_caller_load_settings(mocker, mock_settings):
    mocked_openai_client = mocker.patch(
        "llm_api.backends.openai.OpenaiCaller.get_client"
    )

    caller = OpenaiCaller(mock_settings)
    expected_test_key = mock_settings.openai_api_key.get_secret_value()

    assert caller.settings.openai_api_key.get_secret_value() == expected_test_key
    mocked_openai_client.assert_called_once()


def test_generate_openai_prompt_success():
    user_input = "What day is it today?"

    prompt_output = OpenaiCaller.generate_openai_prompt()
    assert isinstance(prompt_output, ChatPromptTemplate)

    prompt_output = prompt_output.format_messages(text=user_input)

    expected_prompt_elements = 5
    assert len(prompt_output) == expected_prompt_elements
    assert isinstance(prompt_output[-1], HumanMessage)
    assert prompt_output[-1].content == user_input


@pytest.mark.asyncio
async def test_call_model_success(mocker, mock_settings):
    caller = OpenaiCaller(mock_settings)

    expected_entities = ["William Shakespeare", "Globe Theatre"]
    mocked_result_dict = {
        "entities": [
            {
                "uri": "William Shakespeare",
                "description": "English playwright, poet, and actor",
                "wikipedia_url": "https://en.wikipedia.org/wiki/William_Shakespeare",
            },
            {
                "uri": "Globe Theatre",
                "description": "Theatre in London associated with William Shakespeare",
                "wikipedia_url": "https://en.wikipedia.org/wiki/Globe_Theatre",
            },
        ],
        "connections": [
            {
                "from": "William Shakespeare",
                "to": "Globe Theatre",
                "label": "performed plays at",
            },
        ],
        "user_search": "Who is Shakespeare?",
    }
    mocked_result = AIMessage(content=json.dumps(mocked_result_dict))

    mocker.patch(
        "langchain.schema.runnable.base.RunnableSequence.ainvoke",
        return_value=mocked_result,
    )
    user_template = "{text}"
    test_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a test system"),
            ("system", "Provide a valid JSON response to the user."),
            ("user", user_template),
        ]
    )

    test_search = "Who is Shakespeare?"
    response = await caller.call_model(test_prompt, test_search)

    assert expected_entities == [entity["uri"] for entity in response["entities"]]


@pytest.mark.asyncio
async def test_call_model_failure_api_connection_error(mocker, mock_settings):
    caller = OpenaiCaller(mock_settings)

    mocked_client_call = mocker.patch(
        "langchain.schema.runnable.base.RunnableSequence.ainvoke"
    )
    expected_error_message = "Unable to connect to OpenAI. Connection error."
    patched_httpx_request = mocker.patch("httpx.Request")
    mocked_client_call.side_effect = openai.APIConnectionError(
        request=patched_httpx_request
    )

    user_template = "{text}"
    test_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a test system"),
            ("system", "Provide a valid JSON response to the user."),
            ("user", user_template),
        ]
    )
    test_search = "Who is Shakespeare?"

    with pytest.raises(OpenaiModelCallError) as exception:
        await caller.call_model(test_prompt, test_search)
    assert str(exception.value) == expected_error_message


@pytest.mark.asyncio
async def test_call_model_failure_rate_limit_error(mocker, mock_settings):
    caller = OpenaiCaller(mock_settings)

    mocked_client_call = mocker.patch(
        "langchain.schema.runnable.base.RunnableSequence.ainvoke"
    )
    unique_test_error_message = "This is a test case - rate limit error."
    expected_error_message = f"Rate limit exceeded. {unique_test_error_message}"
    mocked_response = mocker.patch("httpx.Response")

    mocked_client_call.side_effect = openai.RateLimitError(
        unique_test_error_message, response=mocked_response, body=None
    )

    user_template = "{text}"
    test_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a test system"),
            ("system", "Provide a valid JSON response to the user."),
            ("user", user_template),
        ]
    )
    test_search = "Who is Shakespeare?"

    with pytest.raises(OpenaiModelCallError) as exception:
        await caller.call_model(test_prompt, test_search)

    assert str(exception.value) == expected_error_message


@pytest.mark.asyncio
async def test_call_model_failure_api_error(mocker, mock_settings):
    caller = OpenaiCaller(mock_settings)

    mocked_client_call = mocker.patch(
        "langchain.schema.runnable.base.RunnableSequence.ainvoke"
    )
    unique_test_error_message = "This is test case - API Error"
    expected_error_message = f"OpenAI API error: {unique_test_error_message}"

    mocked_request = mocker.patch("httpx.Request")

    mocked_client_call.side_effect = openai.APIError(
        unique_test_error_message, request=mocked_request, body=None
    )

    user_template = "{text}"
    test_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a test system"),
            ("system", "Provide a valid JSON response to the user."),
            ("user", user_template),
        ]
    )
    test_search = "Who is Shakespeare?"

    with pytest.raises(OpenaiModelCallError) as exception:
        await caller.call_model(test_prompt, test_search)

    assert str(exception.value) == expected_error_message
