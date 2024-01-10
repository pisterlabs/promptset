"""
Utility functions for API calls

"""
import os
import sys
import openai
import requests
from dotenv import load_dotenv
from typing import Any, Dict
from openai import OpenAI
client = OpenAI()

# load .env file
load_dotenv()

assert os.environ.get("OPENAI_API_KEY")
openai.api_key = os.environ.get("OPENAI_API_KEY")


"""
Purpose:
    Interact with the OpenAI API.
    Provide supporting prompt engineering functions.
"""

def safe_get(data, dot_chained_keys):
    """
    {'a': {'b': [{'c': 1}]}}
    safe_get(data, 'a.b.0.c') -> 1
    """
    keys = dot_chained_keys.split(".")
    for key in keys:
        try:
            if isinstance(data, list):
                data = data[int(key)]
            else:
                data = data[key]
        except (KeyError, TypeError, IndexError):
            return None
    return data


def response_parser(response: Dict[str, Any]):
    return response.choices[0].message.content


# ------------------ content generators ------------------


def prompt(context: str, model: str = "gpt-4-1106-preview") -> str:
    # validate the openai api key - if it's not valid, raise an error
    if not openai.api_key:
        sys.exit(
            """
ERORR: OpenAI API key not found. Please export your key to OPENAI_API_KEY
Example bash command:
    export OPENAI_API_KEY=<your openai apikey>
            """
        )

    prompt = f'''// Goal: Extract useful Q&A pairs from a given text.
Here is the text snippet of interest:
{context}


// Requirements:
// 1. Identify and compile 10 pairs of questions and answers.
// 2. Exclude irrelevant conversational speech.
// 3. Include complete URLs where relevant.
// 4. Omit specific usernames, channels, or timestamps.
// 5. Focus on general-use content, avoiding overly specific user conversations.
// 6. When citing error messages, include the complete error while omitting personal identifiers like IDs or usernames.
// 7. Where relevant, supplement answers with complete code snippets in a code block format.
// Restrictions:
// - Do not invent or create answers; rely solely on the provided text.

// Instructions for AI:
// Analyze the provided text, adhering to the above guidelines, to extract relevant and general Q&A pairs that would be beneficial for a broader audience. Ensure that each answer is clearly connected to its question, maintaining the integrity and context of the original discussion.

// Here are some examples:
```
Question: How do I load an image for processing by an analyzer in Python?
Answer: Use the following code to load an image from a specified file location for processing:
```
user_proxy.initiate_chat(designAnalyzer, message="""Load the image from the <img ./coding/design.jpg> file location for processing by an analyzer""")
```

Question: How do I install a specific version of Autogen with OpenAI?
Answer: To install Autogen 0.2.0b5 with OpenAI 1.2.4, use the command:
```
pip install autogen==0.2.0b5
```
This resolves the issue with the error: InvalidRequestError: Invalid URL (POST /v1/openai/deployments/InnovationGPT4-32/chat/completions) encountered with earlier versions.

Question: How do I integrate Autogen with a project using an older version of OpenAI?
Answer: To integrate Autogen with a project using OpenAI 0.27.4, you might face compatibility issues. Autogen 0.1.14 raises an InvalidRequestError with OpenAI 0.28.1, while Autogen 0.2+ runs fine with OpenAI 1.0+. The compatibility must be checked between specific versions.
```
'''


    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response_parser(response)

