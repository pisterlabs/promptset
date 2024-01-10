import os

import openai
from dotenv import load_dotenv
from icecream import ic
from utils.client import AIDevsClient


# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Get API key from environment variables
aidevs_api_key = os.environ.get("AIDEVS_API_KEY")

# Create a client instance
client = AIDevsClient(aidevs_api_key)

# Define a list of facts
hints = []

for i in range(10):
    # Get a task
    task = client.get_task("whoami")
    ic(task.data)

    # Extract hint
    hint = task.data["hint"]

    # Translate hint
    translation_msg = f"Translate the following from Polish to English (only return the translation and nothing else): {hint}"
    translation = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": translation_msg},
        ],
        max_tokens=100,
    )

    eng_hint = translation.choices[0].message.content
    ic(eng_hint)
    hints.append(eng_hint)

    # Try to figure out "who am I?"
    hints_str = "\n".join([f"- {hint}" for hint in hints])
    whoami_msg = f"""
    Your task is to answer the question "Who am I?".
    To answer this question, you can use the following hints:
    {hints_str}

    If the hints are not enough, just answer with "I don't know" and nothing else.
    """

    whoami = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": whoami_msg},
        ],
        max_tokens=400,
    )

    answer = whoami.choices[0].message.content
    ic(answer)
    if answer != "I don't know":
        # Post an answer
        response = client.post_answer(task, answer)
        ic(response)
        break
