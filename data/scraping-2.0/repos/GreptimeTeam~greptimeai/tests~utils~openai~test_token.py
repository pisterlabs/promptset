from typing import Any, List

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion import ChatCompletion, Choice

from greptimeai.utils.openai.token import (
    extract_chat_inputs,
    extract_chat_outputs,
    get_openai_token_cost_for_model,
    num_tokens_from_messages,
)


def test_num_tokens():
    cases = [
        (
            18,
            "You are a helpful, pattern-following assistant that translates corporate jargon into plain English.",
        ),
        (10, "New synergies will help drive top-line growth."),
        (8, "Things working well together will increase revenue."),
        (
            18,
            "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage.",
        ),
        (15, "Let's talk later when we're less busy about how to do better."),
        (
            19,
            "This late pivot means we don't have time to boil the ocean for the client deliverable.",
        ),
    ]
    for token_count, message in cases:
        assert token_count == num_tokens_from_messages(message)


def test_cal_openai_token_cost_for_model():
    cases = [
        (0.15, ("gpt-3.5-turbo-0613", 100000, False)),
        (0.2, ("gpt-3.5-turbo-0613", 100000, True)),
        (0, ("unknown", 10, False)),
        (0, ("unknown", 10, True)),
        (0.04, ("text-embedding-ada-002", 100000, False)),
    ]
    for cost, args in cases:
        model, num, is_completion = args
        assert cost == get_openai_token_cost_for_model(model, num, is_completion)


def test_extract_chat_inputs():
    messages: List[Any] = [
        ChatCompletionUserMessageParam(role="user", content="Hello"),
        ChatCompletionAssistantMessageParam(
            role="assistant", content="Hi, how can I help you?"
        ),
        ChatCompletionUserMessageParam(role="user", content="I have a question."),
        ChatCompletionAssistantMessageParam(
            role="assistant", content="Sure, what's your question?"
        ),
    ]
    expected_output = "user: Hello\nassistant: Hi, how can I help you?\nuser: I have a question.\nassistant: Sure, what's your question?"
    assert extract_chat_inputs(messages) == expected_output


def test_extract_chat_outputs():
    completion = ChatCompletion(
        id="test_id",
        object="chat.completion",
        created=1678901234,
        model="gpt-3.5-turbo-0613",
        choices=[
            Choice(
                message=ChatCompletionMessage(
                    role="assistant", content="Here is the answer:"
                ),
                logprobs=None,
                finish_reason="stop",
                index=0,
            ),
            Choice(
                message=ChatCompletionMessage(role="assistant", content="Beijing"),
                finish_reason="stop",
                logprobs=None,
                index=1,
            ),
        ],
    )
    expected_output = "assistant: Here is the answer:\nassistant: Beijing"
    assert extract_chat_outputs(completion.model_dump()) == expected_output
