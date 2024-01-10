import inspect
import os
import sys
import traceback

import openai
from openai import ChatCompletion
from openai.error import OpenAIError

from rich.console import Console
from rich.markdown import Markdown


openai.api_key = os.getenv("OPENAPI_API_KEY")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_PROMPT = os.getenv(
    "OPENAI_PROMTP",
    """
    You are a chatbot, act as an instructor, teaching errors in Python code to beginners.
    I will give you code and exceptions and you will provide explanations and solutions.
    Reply in Markdown with Python code blocks.
    """
)
RESULT_MESSAGE = """
# Code

```python
{code}
```

# Exception

```python
{exception_stack}
{exception_type}: {exception}
```

# Details

{result}
"""

PROMPT_INPUT = """
{code}

{exception_stack}
{exception_type}: {exception}
"""


def chat_exception_hook(exc_type, exc_value, exc_traceback):
    """Exception hook.

    Args:
        exc_type (type): Exception type
        exc_value (Exception): Exception instance
        exc_traceback (Traceback): Traceback object
    """
    stack_call = "".join(traceback.format_tb(exc_traceback))
    source_code = inspect.getsource(exc_traceback)

    code_found = False
    lines = []
    for line in source_code.split("\n"):
        if line == "import pychaterr" or line.startswith("from pychaterr"):
            continue
        code_found = code_found or bool(line.strip())
        if not code_found and line.strip() == "":
            continue
        lines.append(line)

    code = "\n".join(lines)

    try:
        response = ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": OPENAI_PROMPT,
                },
                {
                    "role": "user",
                    "content": PROMPT_INPUT.format(
                        code=code,
                        exception_stack=stack_call,
                        exception_type=exc_type.__name__,
                        exception=exc_value,
                    )
                }
            ],
        )
    except OpenAIError as error:
        print("OpenAI API error: \"{error}\"".format(error=error))
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    else:
        Console().print(
            Markdown(
                RESULT_MESSAGE.format(
                    code=code,
                    exception_stack=stack_call,
                    exception_type=exc_type.__name__,
                    exception=exc_value,
                    result="".join([
                        choice.message.content
                        for choice in response.choices
                    ])
                )
            )
        )


def setup_handler():
    """Setup general exception handler and enable ChatGTP error processing."""
    if not os.getenv("PYCHATERR_DISABLED") == "yes":
        if sys.excepthook != chat_exception_hook:
            sys.excepthook = chat_exception_hook