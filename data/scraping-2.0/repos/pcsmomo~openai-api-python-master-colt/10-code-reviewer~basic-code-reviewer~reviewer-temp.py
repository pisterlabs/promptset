import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


PROMPT = """
You will receive a file's contents as text.
Generate a code review for the file.  Indicate what changes should be made to improve its style, performance, readability, and maintainability.  If there are any reputable libraries that could be introduced to improve the code, suggest them.  Be kind and constructive.  For each suggested change, include line numbers to which you are referring
"""

filecontent = """
def mystery(x, y):
    return x ** y
"""

messages = [
    {"role": "system", "content": PROMPT},
    {"role": "user", "content": f"Code review the following file: {filecontent}"}
]

res = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages
)

print(res["choices"][0]["message"])

'''
This file is a single line function definition with clear input and output. There are some suggestions that could be addressed to improve its quality:

1. Add docstring: Although this function takes only two arguments and the return output is almost self-explanatory. It's always a good practice to have a docstring that clearly describes what the function does, what the input parameter types should be, and what the output type/format should be. This can help anyone who is using the function to quickly understand what the function does and how to use it.

2. Rename function and input parameters: The function name "mystery" and the input parameter names "x" and "y" doesn't convey any meaning as to what the function does. Renaming the function to something like "power" and input parameters to "base" and "exponent" would be clearer and more descriptive.

3. Enclose math operations in parentheses: Although the expression in the return statement is mathematically correct, it is recommended to explicitly enclose math operations in parentheses to avoid any ambiguity in their order of execution.

4. Add type hints: Adding type hints to input and output can improve readability and maintainability of the code.

Here is an updated version of the code with these suggestions applied:

```
def power(base: int, exponent: int) -> int:
    """
    Compute the power of a given base.

    Args:
      base: The base value (integer) for which power needs to be computed
      exponent: The exponent value (integer) for which power needs to be computed

    Returns:
      integer that represents the result of the base raised with the exponent.
    """
    return (base ** exponent)
```

I hope these suggestions will help you improve the code's quality.
'''
