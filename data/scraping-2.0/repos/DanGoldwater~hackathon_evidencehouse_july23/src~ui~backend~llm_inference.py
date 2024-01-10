import os
import openai
import textwrap
from .consts import MODEL_TYPE
from dotenv import load_dotenv
from typing import Generator, Union

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")


def load_system_prompt() -> str:
    return "You are a data analyst in charge of assisting the UK government with procurement of contracts"


def load_intial_prompt(input_description: str, previous_jobs: str) -> str:
    prompt = textwrap.dedent(
        f"""
The UK government is doing a procurement of {input_description}. 
Please give me a list of bullet points with key drivers of costs and key risks with this type of procurement
You should be extremely specific and detailed in your answers.

Some related, previous project descriptions are:
{previous_jobs}

Return your answer in the format:
You must return estimated values for each cost driver and risk factor.
Overview:
...

Cost Drivers:
- Title: {{title}}
  Description: {{description}}
  MinCost (GBP): {{minCost}}
  MaxCost (GBP): {{maxCost}}

Risks:
- Title: {{title}}
  Description: {{description}}
  Likelihood: {{likelihood}}
  Impact: {{impact}}
  MinCost (GBP): {{minCost}}
  MaxCost (GBP): {{maxCost}}
...

Summary:
...
"""
    )
    return prompt


def create_chat_completion(
    messages: list[dict[str, str]],
) -> Union[Generator, str]:
    response = openai.ChatCompletion.create(
        messages=messages,
        model=MODEL_TYPE,
        stream=True,
    )
    for chunk in response:
        if len(chunk["choices"][0]["delta"]) != 0:
            yield chunk["choices"][0]["delta"]["content"]
