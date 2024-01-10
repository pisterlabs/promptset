# llm_call.py
from langchain.llms import OpenAI

# --------------------------------------------------------------
# LLMs: Get predictions from a language model
# --------------------------------------------------------------


def run_llm_demo():
    llm = OpenAI(model_name="text-davinci-003")
    prompt = "Write a poem about python and ai"
    return llm(prompt)
