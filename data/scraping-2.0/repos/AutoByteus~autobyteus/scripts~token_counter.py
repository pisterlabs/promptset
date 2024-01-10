import asyncio

from langchain.callbacks import get_openai_callback
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
with get_openai_callback() as cb:
    llm("What is the square root of 4?")

total_tokens = cb.total_tokens
assert total_tokens > 0

with get_openai_callback() as cb:
    llm("What is the square root of 4?")
    llm("What is the square root of 4?")

assert cb.total_tokens == total_tokens * 2