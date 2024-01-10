import re
import openai
from loguru import logger
from clipboard import set_clipboard
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")


def extract_python_code(md_string):
    pattern = r"```python\n(.*?)\n```"
    matches = re.findall(pattern, md_string, re.DOTALL)

    if matches:
        return matches[0].strip()

    return md_string


def ask(question):
    """Send a question to ChatGPT and get the response."""
    message = [{"role": "user", "content": question}]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=message,
        max_tokens=2000,  # Adjust as necessary
    )
    return response.choices[0].message.content.strip()


def send_to_llm(content):
    logger.info("Sending to LLM")
    resp = ask(content)
    logger.info("Got response")
    print(resp)
    resp = extract_python_code(resp)
    set_clipboard(resp)
