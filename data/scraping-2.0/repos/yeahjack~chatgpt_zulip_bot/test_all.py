import pytest
from chatgpt import OpenAI, prompt_manager
from configparser import ConfigParser
import tiktoken
import os

api_version = os.environ.get("API_VERSION")
api_key = os.environ.get("OPENAI_API_KEY")


@pytest.fixture
# Test for chatgpt.py
def openai_object():
    return OpenAI(api_version=api_version, api_key=api_key)


def test_trim_conversation_history(openai_object):
    def count_tokens(message, api_version):
        if "gpt-3.5-turbo" in api_version:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        elif "gpt-4" in api_version:
            encoding = tiktoken.encoding_for_model("gpt-4")
        else:
            raise ValueError("Unsupported API version, your API version is: " + api_version)
        return len(encoding.encode(message))

    history = [
        "User: Hello",
        "AI: Hi there!",
        "User: How are you?",
        "AI: I'm doing great! How about you?",
        "User: I'm fine, thank you."
    ]

    # Calculate token count for each message
    message_tokens = [count_tokens(
        msg, openai_object.api_version) for msg in history]

    # Set max_tokens to the sum of tokens of the last two messages
    max_tokens = message_tokens[-1] + message_tokens[-2]

    trimmed_history = openai_object.trim_conversation_history(
        history, max_tokens)
    assert len(
        trimmed_history) == 2, "Trimmed history should have only the last two messages"


def test_chatgpt_response_strings(openai_object):
    prompts = [
        "/1",
        "/end",
        "hi",
        "/polish_en this is a test cases",
        "/polish_zh 测试用例这一个是。",
        "/find_grammar_mistakes this is a test cases",
        "/zh-en 这是一个测试用例",
        "/en-zh This is a test case",
        "/en-ac This is a test case",
        "/ex_code_zh print('This is a test case.')"
    ]

    try:
        for prompt in prompts:
            result = openai_object.get_chatgpt_response(0, prompt)
            assert isinstance(
                result, str), f"Result should be a string for prompt: {prompt}"
    finally:
        # Clear the conversation history for test cases.
        openai_object.user_conversations[0] = []

def test_prompt_manager():
    # incorrect command
    msg, code = prompt_manager("/zh-end asdf")
    assert msg == 'Sorry, command not found: `/zh-end`, type `/help` to get the list of commands.'
    assert code == 2

    # correct command
    msg, code = prompt_manager("/zh-en asdf")
    assert msg == 'As a translator, your task is to accurately translate text from Chinese to English. Please pay attention to context and accurately explain phrases and proverbs. Below is the text you need to translate: \n\nasdf'
    assert code == 0
    
    # contextual text information
    msg, code = prompt_manager("zh-en asdf")
    assert msg == 'zh-en asdf'
    assert code == 1
