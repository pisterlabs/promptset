"""
tests/conftest.py
"""
import json
import os
from typing import List, Union

import pytest

from pygptprompt.config.manager import ConfigurationManager
from pygptprompt.json.base import JSONBaseTemplate
from pygptprompt.json.list import JSONListTemplate
from pygptprompt.json.mapping import JSONMappingTemplate
from pygptprompt.model.base import ChatModel, ChatModelResponse
from pygptprompt.model.factory import ChatModelFactory
from pygptprompt.model.llama_cpp import LlamaCppModel
from pygptprompt.model.openai import OpenAIModel

# from pygptprompt.session.token import ChatSessionTokenManager

# from pygptprompt.session.model import SessionModel
# from pygptprompt.session.policy import SessionPolicy


def pytest_addoption(parser):
    parser.addoption(
        "--private",
        action="store_true",
        default=False,
        help="test private endpoints",
    )
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="run slow tests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "private: test private endpoints")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--private"):
        # --no-skip given: do not skip tests
        return
    private = pytest.mark.skip(reason="need --private option to run")
    for item in items:
        if "private" in item.keywords:
            item.add_marker(private)


@pytest.fixture
def encoding_input() -> Union[str, List[str]]:
    return "This is a test sentence."


@pytest.fixture
def embedding_input() -> Union[str, List[str]]:
    return "This is a test sentence."


@pytest.fixture(scope="module")
def message() -> ChatModelResponse:
    return ChatModelResponse(
        role="user",
        content="How many cards are there in a single deck of cards?",
    )


@pytest.fixture(scope="module")
def messages() -> List[ChatModelResponse]:
    return [
        ChatModelResponse(role="system", content="You are a helpful assistant."),
        ChatModelResponse(role="user", content="Who won the world series in 2020?"),
        ChatModelResponse(
            role="assistant",
            content="The Los Angeles Dodgers won the World Series in 2020.",
        ),
    ]


@pytest.fixture(scope="module")
def chat_completion() -> List[ChatModelResponse]:
    return [
        ChatModelResponse(role="system", content="You are a helpful assistant."),
        ChatModelResponse(
            role="user",
            content="Translate the following English text to French: 'The quick brown fox jumped over the lazy dog.'",
        ),
    ]


@pytest.fixture(scope="module")
def function_completion() -> List[ChatModelResponse]:
    return [
        ChatModelResponse(role="system", content="You are a helpful assistant."),
        ChatModelResponse(
            role="user", content="What's the weather like in San Francisco, CA?"
        ),
    ]


@pytest.fixture
def mock_weather_callback() -> object:
    """
    A mock function for getting the current weather.
    """

    def get_current_weather(location: str, unit: str = "celsius") -> str:
        """
        Get the current weather in a given location.

        Returns:
            str: A string that describes the current weather.
        NOTE:
            This is a mock function, so let's return a mock weather report.
        """
        return f"The current weather in {location} is 20 degrees {unit}."

    return get_current_weather


@pytest.fixture(scope="module")
def temp_json_path() -> str:
    return "tests/test.temp.json"


@pytest.fixture(scope="module")
def temp_json_backup_path() -> str:
    return "tests/test.temp.backup.json"


@pytest.fixture(scope="module")
def temp_json_nested_path() -> str:
    return "tests/nested/test.temp.json"


@pytest.fixture(scope="module")
def temp_json_data() -> dict:
    return {"test": "data"}


@pytest.fixture
def temp_json_file(temp_json_path: str, temp_json_data: dict):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(temp_json_path), exist_ok=True)

    # Write the temp data to the file
    with open(temp_json_path, "w") as f:
        json.dump(temp_json_data, f)

    yield temp_json_path

    # Cleanup the temporary file after the test
    os.remove(temp_json_path)


@pytest.fixture(scope="module")
def json_base_template(temp_json_path: str, temp_json_data: dict) -> JSONBaseTemplate:
    return JSONBaseTemplate(temp_json_path, temp_json_data)


@pytest.fixture(scope="module")
def json_base_template_nested(
    temp_json_nested_path: str, temp_json_data: dict
) -> JSONBaseTemplate:
    return JSONBaseTemplate(temp_json_nested_path, temp_json_data)


@pytest.fixture
def temp_json_list(
    temp_json_path,
    messages: List[ChatModelResponse],
):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(temp_json_path), exist_ok=True)

    # Write the temp data to the file
    with open(temp_json_path, "w") as f:
        json.dump(messages, f)

    yield temp_json_path

    # Cleanup the temporary file after the test
    os.remove(temp_json_path)


@pytest.fixture(scope="function")
def json_list_template(
    temp_json_path: str,
    messages: List[ChatModelResponse],
) -> JSONListTemplate:
    return JSONListTemplate(temp_json_path, messages)


@pytest.fixture(scope="module")
def temp_map_data() -> dict:
    return {"key1": "value1", "nested": {"key2": "value2"}}


@pytest.fixture
def temp_json_map(temp_json_path: str, temp_map_data: dict):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(temp_json_path), exist_ok=True)

    # Write the temp data to the file
    with open(temp_json_path, "w") as f:
        json.dump(messages, f)

    yield temp_json_path

    # Cleanup the temporary file after the test
    os.remove(temp_json_path)


@pytest.fixture(scope="module")
def json_map_template(
    temp_json_path: str,
    temp_map_data: dict,
) -> JSONMappingTemplate:
    return JSONMappingTemplate(temp_json_path, temp_map_data)


@pytest.fixture(scope="module")
def config_file_path() -> str:
    if os.path.exists("config.json"):
        return "config.json"
    return "tests/config.dev.json"


@pytest.fixture(scope="module")
def config(config_file_path: str) -> ConfigurationManager:
    return ConfigurationManager(config_file_path)


@pytest.fixture(scope="module")
def openai_model(config: ConfigurationManager) -> OpenAIModel:
    return OpenAIModel(config=config)


@pytest.fixture(scope="module")
def llama_cpp_model(config: ConfigurationManager) -> LlamaCppModel:
    return LlamaCppModel(config=config)


@pytest.fixture(scope="module")
def chat_model_factory(config: ConfigurationManager):
    return ChatModelFactory(config=config)


@pytest.fixture(scope="module")
def chat_model(chat_model_factory: ChatModelFactory) -> ChatModel:
    return chat_model_factory.create_model(provider="llama_cpp")


# @pytest.fixture(scope="module")
# def token_manager(
#     config: ConfigurationManager, chat_model: ChatModel
# ) -> ChatSessionTokenManager:
#     return ChatSessionTokenManager(
#         provider="llama_cpp",
#         config=config,
#         chat_model=chat_model,
#     )


# @pytest.fixture(scope="module")
# def session_model(config: ConfigurationManager) -> SessionModel:
#     return SessionModel(config)


# @pytest.fixture(scope="module")
# def session_policy(config):
#     return SessionPolicy(config)
