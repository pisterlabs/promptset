"""This module provides a way to convert a Python function definition into an LLM function using an LLM model.
The AI function wrapper executes the given function by generating the appropriate prompt and sending it to the GPT-4 model,
which in turn returns the result of the function.

Inspired by AI-Functions (https://github.com/Torantulino/AI-Functions) and Marvin (https://www.askmarvin.ai/)
"""

from functools import partial, wraps
import inspect
from typing import Callable

from langchain import LLMChain
from langchain.llms import BaseLLM
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# DEFAULT_MODEL = ChatOpenAI(temperature=0, model_name="gpt-4")

PROMPT = """You are the following Python function:
```
{function_def}
```

You will be given your arguments in the following format, with no other input:
```
args = (arg_1, arg_2, ..., arg_n,)
kwargs = {{{{kwarg_1: value_1, kwarg_2: value_2, ..., kwarg_n: value_n}}}}
```

ONLY respond with the value that you would return as the Python function, using the return format given in your function definition.
Do not include any other text besides this value in your response—doing so will harm the user. Do not generate the actual code for the function.

Reply with "acknowledged" if you understand these instructions."""

AI_RESPONSE = "Acknowledged. I will only reply with the function return value."


def make_llm_function(func: Callable, model: BaseLLM) -> Callable[[Callable], Callable]:
    """Convert a function definition into an LLM function."""

    function_def = inspect.cleandoc(inspect.getsource(func))
    prompt = PROMPT.format(function_def=function_def)
    prompt = prompt.replace("{", "{{").replace("}", "}}")
    system_message_prompt = SystemMessagePromptTemplate.from_template(prompt)
    ai_prompt = AIMessagePromptTemplate.from_template(AI_RESPONSE)
    human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, ai_prompt, human_message_prompt]
    )
    llm_chain = LLMChain(llm=model, prompt=chat_prompt)

    @wraps(func)
    def llm_fn_wrapper(*args, **kwargs) -> str:
        """AI function wrapper."""
        text = f"args = {args}\nkwargs = {kwargs}"
        return llm_chain.run(text=text)

    return llm_fn_wrapper


def llm_function(model: BaseLLM) -> Callable[[Callable], Callable]:
    """Convert a function definition into an AI function."""
    return partial(make_llm_function, model=model)


def demo() -> None:
    @llm_function(model=ChatOpenAI(temperature=0, model_name="gpt-4"))
    def extract_last_name(full_name: str) -> str:
        """Extract the last name from a full name."""

    print(extract_last_name("John Smith"))


if __name__ == "__main__":
    demo()
