import openai
import pytest
import tiktoken

from src.models.openai_model import (
    OpenAIChatModels,
    OpenAITextModels,
    generate_chat_completion,
    generate_response_with_turns,
    generate_text_completion,
    generate_text_completion_with_logprobs,
)


def test_all_openai_models_found():
    ours = set(OpenAIChatModels.list() + OpenAITextModels.list())
    theirs = {m["id"] for m in openai.Model.list().data}
    missing = ours - theirs
    assert not missing, f"Missing models: {missing}"


@pytest.mark.parametrize("model", OpenAITextModels.list())
def test_generate_text_completion(model):
    test_prompt = "Once upon a time,"
    max_tokens = 5
    text = generate_text_completion(test_prompt, model=model, max_tokens=max_tokens)
    tokens = tiktoken.encoding_for_model(model).encode(text)
    assert len(tokens) == max_tokens
    assert (
        len(set(tokens)) == max_tokens
    )  # sanity check: assume all tokens are unique (reasonable for short text)


@pytest.mark.parametrize("model", OpenAITextModels.list())
def test_generate_text_completion_with_logprobs(model):
    prompt = "It was a bright cold day in April,"
    max_tokens = 4
    tokens, logprobs = generate_text_completion_with_logprobs(
        prompt, model=model, max_tokens=max_tokens, echo=False
    )
    assert len(tokens) == max_tokens
    assert len(tokens) == len(logprobs)
    assert (
        len(set(tokens)) == max_tokens
    )  # sanity check: assume all tokens are unique (reasonable for short text)

    # echo=True
    tokens, logprobs = generate_text_completion_with_logprobs(
        prompt, model=model, max_tokens=max_tokens, echo=True
    )
    expected_tokens = max_tokens + len(
        tiktoken.encoding_for_model(model).encode(prompt)
    )
    assert len(tokens) == expected_tokens
    assert len(tokens) == len(logprobs)
    assert (
        len(set(tokens)) == expected_tokens
    )  # sanity check: assume all tokens are unique (reasonable for short text)


@pytest.mark.parametrize("model", OpenAIChatModels.list())
def test_generate_chat_completion(model):
    max_tokens = 5
    test_prompt = [
        {"role": "user", "content": "Hello, how are you?"},
    ]
    text = generate_chat_completion(test_prompt, model=model, max_tokens=max_tokens)
    tokens = tiktoken.encoding_for_model(model).encode(text)
    assert len(tokens) == max_tokens
    assert (
        len(set(tokens)) == max_tokens
    )  # sanity check: assume all tokens are unique (reasonable for short text)


@pytest.mark.parametrize("model", OpenAITextModels.list() + OpenAIChatModels.list())
def test_generate_response_with_turns(model):
    max_tokens = 3
    turns = [
        {"role": "user", "content": "Tell me a joke!"},
        {"role": "assistant", "content": "Why did the chicken cross the road?"},
        {
            "role": "user",
            "content": "I don't know, why did the chicken cross the road?",
        },
    ]
    text = generate_response_with_turns(
        model=model,
        turns=turns,
        max_tokens=max_tokens,
    )
    tokens = tiktoken.encoding_for_model(model).encode(text)
    assert len(tokens) == max_tokens
    assert (
        len(set(tokens)) == max_tokens
    )  # sanity check: assume all tokens are unique (reasonable for short text)
