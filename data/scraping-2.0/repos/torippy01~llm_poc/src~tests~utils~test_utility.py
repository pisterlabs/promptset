import openai
import langchain
import pytest

from langchain.chat_models import ChatOpenAI
from mdutils import MdUtils

from xecretary_core.utils.utility import (
    set_up,
    get_gpt_response,
    create_llm,
    create_CBmemory,
    sep_md,
    host_validation,
    port_validation,
)


@pytest.fixture
def openai_response():
    return {
        "id": "chatcmpl-8LnfWQnWNQJ1P3BXXnr2f9e5AULjc",
        "object": "chat.completion",
        "created": 1700206506,
        "model": "gpt-3.5-turbo-0613",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I assist you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 8, "completion_tokens": 9, "total_tokens": 17},
    }


@pytest.fixture
def chat_completion_create_mock(mocker, openai_response):
    return mocker.patch.object(
        openai.ChatCompletion, "create", return_value=openai_response
    )


def test_set_up(mocker):
    load_dotenv_mock = mocker.patch("xecretary_core.utils.utility.load_dotenv")
    set_up()
    load_dotenv_mock.assert_called_once()
    assert langchain.verbose


def test_get_gpt_response(chat_completion_create_mock):
    expect = get_gpt_response("Hello")
    answer = "Hello! How can I assist you today?"
    assert expect == answer
    chat_completion_create_mock.assert_called_once()


def test_create_llm():
    llm = create_llm("gpt-4")
    assert isinstance(llm, ChatOpenAI)


def test_sep_md(tmp_path):
    md_filepath = tmp_path / "test.md"
    md_file = MdUtils(md_filepath)
    sep_md(md_file)
    answer = "\n\n\n  \n  \n---  \n"
    expect = md_file.get_md_text()
    assert expect == answer


def test_host_validation():
    assert host_validation("localhost")
    assert not host_validation(None)
    assert not host_validation(100)


def test_port_validation():
    assert port_validation("3999")
    assert not port_validation(None)
    assert not port_validation("test3999")
    assert not port_validation("test")
