# All prompts are loaded through the `load_prompt` function.
from langchain.prompts import load_prompt


def test_json():
    prompt = load_prompt("../../json/summarization/summarization.json")
    print(prompt.format(documents="funny", require="chickens"))
