import pinecone

from .tool import Tool

from openai import OpenAI
from openai.types.chat import ChatCompletion
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.local"))


client = OpenAI()

function_description = {
    "name": "write_prd",
    "description": "This tool is used to write requirements and pseudocode for the zkApp.",
    "parameters": {},
}

function_messages = "Preparing Requirements and Pseudocode for zkApp"

SYSTEM_PROMPT = """
# Output Format
## Requirements
* Requirement 1
...
* Requirement n

## Pseudocode
[Pseudocode here]
""".strip()


def run_tool(history):
    """Creates a prd using chatgpt"""
    chat_completion: ChatCompletion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
        ],
        model="gpt-4-1106-preview",
    )
    return chat_completion.choices[0].message.content


prd_tool = Tool(
    name="write_prd",
    description=function_description,
    message=function_messages,
    function=run_tool,
)
