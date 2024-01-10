import json

import pytest
from langchain.prompts import ChatPromptTemplate
from langchain.schema.exceptions import LangChainException
from langchain.schema.messages import HumanMessage

from llm_api.backends.bedrock import BedrockCaller, BedrockModelCallError

pytest_plugins = ("pytest_asyncio",)


def test_bedrock_caller_load_settings(mocker, mock_settings):
    mocked_boto3_client = mocker.patch(
        "llm_api.backends.bedrock.BedrockCaller.get_boto3_client"
    )
    mocked_bedrock_client = mocker.patch(
        "llm_api.backends.bedrock.BedrockCaller.get_client"
    )

    caller = BedrockCaller(mock_settings)
    expected_test_key = mock_settings.aws_secret_access_key.get_secret_value()

    assert caller.settings.aws_secret_access_key.get_secret_value() == expected_test_key
    mocked_boto3_client.assert_called_once()
    mocked_bedrock_client.assert_called_once()


def test_generate_openai_prompt_success():
    user_input = "What day is it today?"

    prompt_output = BedrockCaller.generate_prompt()
    assert isinstance(prompt_output, ChatPromptTemplate)

    prompt_output = prompt_output.format_messages(text=user_input)

    expected_prompt_elements = 5
    assert len(prompt_output) == expected_prompt_elements
    assert isinstance(prompt_output[-1], HumanMessage)
    assert prompt_output[-1].content == user_input


@pytest.mark.asyncio
async def test_call_model_success(mocker, mock_settings):
    caller = BedrockCaller(mock_settings)

    expected_entities = ["William Shakespeare", "Globe Theatre"]
    mocked_result = {
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
    mocked_result = "Test json ```json" + json.dumps(mocked_result) + "```"
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
async def test_call_model_failure_index_error(mocker, mock_settings):
    caller = BedrockCaller(mock_settings)
    mocked_result = {
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
    mocked_result = "Test json" + json.dumps(mocked_result)
    mocker.patch(
        "langchain.schema.runnable.base.RunnableSequence.ainvoke",
        return_value=mocked_result,
    )
    expected_error_message = "Unable to parse model output as expected."

    user_template = "{text}"
    test_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a test system"),
            ("system", "Provide a valid JSON response to the user."),
            ("user", user_template),
        ]
    )
    test_search = "Who is Shakespeare?"

    with pytest.raises(BedrockModelCallError) as exception:
        await caller.call_model(test_prompt, test_search)
    assert expected_error_message in str(exception.value)


@pytest.mark.asyncio
async def test_call_model_failure_json_decode_error(mocker, mock_settings):
    caller = BedrockCaller(mock_settings)
    mocked_result = {
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
    mocked_result = "Test ```json," + json.dumps(mocked_result) + "```"
    mocker.patch(
        "langchain.schema.runnable.base.RunnableSequence.ainvoke",
        return_value=mocked_result,
    )
    expected_error_message = "Error decoding model output."

    user_template = "{text}"
    test_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a test system"),
            ("system", "Provide a valid JSON response to the user."),
            ("user", user_template),
        ]
    )
    test_search = "Who is Shakespeare?"

    with pytest.raises(BedrockModelCallError) as exception:
        await caller.call_model(test_prompt, test_search)
    assert expected_error_message in str(exception.value)


@pytest.mark.asyncio
async def test_call_model_failure_api_error(mocker, mock_settings):
    caller = BedrockCaller(mock_settings)

    mocked_client_call = mocker.patch(
        "langchain.schema.runnable.base.RunnableSequence.ainvoke"
    )
    expected_error_message = "Error calling model."

    mocked_client_call.side_effect = ValueError()

    user_template = "{text}"
    test_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a test system"),
            ("system", "Provide a valid JSON response to the user."),
            ("user", user_template),
        ]
    )
    test_search = "Who is Shakespeare?"

    with pytest.raises(BedrockModelCallError) as exception:
        await caller.call_model(test_prompt, test_search)

    assert expected_error_message in str(exception.value)


@pytest.mark.asyncio
async def test_call_model_failure_langchain_error(mocker, mock_settings):
    caller = BedrockCaller(mock_settings)

    mocked_client_call = mocker.patch(
        "langchain.schema.runnable.base.RunnableSequence.ainvoke"
    )
    expected_error_message = "Error sending prompt to LLM."

    mocked_client_call.side_effect = LangChainException()

    user_template = "{text}"
    test_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a test system"),
            ("system", "Provide a valid JSON response to the user."),
            ("user", user_template),
        ]
    )
    test_search = "Who is Shakespeare?"

    with pytest.raises(BedrockModelCallError) as exception:
        await caller.call_model(test_prompt, test_search)

    assert expected_error_message in str(exception.value)
