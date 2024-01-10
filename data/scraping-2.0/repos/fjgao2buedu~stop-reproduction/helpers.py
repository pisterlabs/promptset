import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)
prompt_count = 0
role = "You are an expert computer science researcher and programmer, especially skilled at optimizing algorithms."
debug = False

def prompt(message, temperature):
    global prompt_count
    print("helpers: [PROMPTING NEW SOLUTIONS]")
    prompt_count += 1
    if not debug:
        response = client.chat.completions.create(
            model="gpt-4-0314", 
            response_format={"type": "text"},
            messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": message},
            ],
            temperature=temperature,
        )
        print(response.choices[0].message.content)
        print(f"helpers: [PROMPTING {prompt_count} SUCCESSFUL]")
        return response.choices
    raise
    response_choices = ["```pythondef algorithm(a): return sorted(a)```"]

    # print(response.choices[0].message.content)
    print(f"helpers: [PROMPTING {prompt_count} SUCCESSFUL]")
    return response_choices


def extract_code(responses):
    print("helpers: [EXTRACTING CODE.....]")
    """
    Extracts code snippets from a batch of text.
    This example assumes Python code enclosed in triple backticks for simplicity.
    """

    solutions = []
    pattern = re.compile(r"```python(.*)```", re.DOTALL)  # Regex pattern to extract Python code blocks
    if not debug:
        response = str(responses[0].message.content)
    else:
        response = str(responses[0])
    matches = re.findall(pattern, response)

    solutions.append(matches[0])
    print("helpers: [EXTRACTING CODE SUCCESSFUL]")
    return solutions
