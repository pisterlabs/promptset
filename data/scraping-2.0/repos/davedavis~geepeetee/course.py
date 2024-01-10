"""This is just basic scratch of GPT API usage working through their
prompt course.
"""

import openai
import os
from dotenv import load_dotenv

from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import TerminalFormatter


def highlight_code(code, language):
    lexer = get_lexer_by_name(language)
    formatter = TerminalFormatter()
    highlighted_code = highlight(code, lexer, formatter)
    return highlighted_code


# _ = load_dotenv(find_dotenv())
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_completion(prompt, model="gpt-4"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
        # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


# text = f"""
# You should express what you want a model to do by \
# providing instructions that are as clear and \
# specific as you can possibly make them. \
# This will guide the model towards the desired output, \
# and reduce the chances of receiving irrelevant \
# or incorrect responses. Don't confuse writing a \
# clear prompt with writing a short prompt. \
# In many cases, longer prompts provide more clarity \
# and context for the model, which can lead to \
# more detailed and relevant outputs.
# """
# prompt = f"""
# Summarize the text delimited by triple backticks \
# into a single sentence.
# ```{text}```
# """
# response = get_completion(prompt)
# print(response)
#
prompt = """
Write me a python class that queries a BigQuery table for sales by week.
"""
response = get_completion(prompt)
print(highlight_code(response, language="python"))
