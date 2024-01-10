import openai
import pytest
import yaml
import os


@pytest.fixture
def is_tested():
    if not os.path.exists("data/gpt_info.yaml"):
        return False
    with open("data/gpt_info.yaml", "r") as f:
        gpt_info = yaml.safe_load(f)
    return gpt_info["test_gpt_part"]


@pytest.fixture
def get_gpt_info():
    if not os.path.exists("data/gpt_info.yaml"):
        return None
    with open("data/gpt_info.yaml", "r") as f:
        gpt_info = yaml.safe_load(f)
    return gpt_info["OpenAI"]


@pytest.fixture
def get_gpt_parameters():
    if not os.path.exists("data/gpt_info.yaml"):
        return None
    with open("data/gpt_info.yaml", "r") as f:
        gpt_info = yaml.safe_load(f)
    return gpt_info["gpt_params"]
