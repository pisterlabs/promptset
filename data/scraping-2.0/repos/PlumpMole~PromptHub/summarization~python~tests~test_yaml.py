# All prompts are loaded through the `load_prompt` function.
from langchain.prompts import load_prompt

def test_yaml():
    prompt = load_prompt("../../yaml/summarization/summarization.yaml")
    print(prompt.format(documents="funny", require="chickens"))
