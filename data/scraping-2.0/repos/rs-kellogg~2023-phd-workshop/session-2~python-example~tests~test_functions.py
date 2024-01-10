import pytest
import openai
import openaihelper.functions as F


def test_config(config):
    keys = [
        "prompt",
        "encoding_name",
        "model_name",
        "max_token_len",
        "openai_api_key_file",
    ]
    for key in keys:
        assert key in config


def test_key(openai_api_key):
    assert len(openai_api_key) == 51
    openai.api_key = openai_api_key


def test_sample1(sample_1):
    assert len(sample_1) == 1


def test_sample100(sample_100):
    assert len(sample_100) == 100


def test_count_tokens(config, sample_100):
    prompt = config["prompt"]
    encoding_name = config["encoding_name"]
    max_token_len = config["max_token_len"]

    texts = list(sample_100["text"])
    num_prompt_tokens = F.count_tokens(prompt, encoding_name)
    assert num_prompt_tokens == 163
    for idx, text in enumerate(texts):
        num_tokens = F.count_tokens(f"{prompt} {text}", encoding_name)
        assert num_tokens > 0
        if idx not in [31, 74, 99]:
            assert num_tokens < max_token_len
        else:
            assert num_tokens > max_token_len


@pytest.mark.skipif(
    "not config.getoption('--use-api')",
    reason="Only run when --use-api is given",
)
def test_chat_complete_success_mode(config, sample_100):
    prompt = config["prompt"]
    model_name = config["model_name"]
    text = list(sample_100["text"])[0]
    response = F.chat_complete(text, model_name, prompt)
    assert type(response) == openai.openai_object.OpenAIObject
    print(str(response))


@pytest.mark.skipif(
    "not config.getoption('--use-api')",
    reason="Only run when --use-api is given",
)
def test_chat_complete_failure_mode(config, sample_100):
    prompt = config["prompt"]
    model_name = config["model_name"]
    text = list(sample_100["text"])[99]
    response = F.chat_complete(text, model_name, prompt)
    assert type(response) == str
    print(response)
