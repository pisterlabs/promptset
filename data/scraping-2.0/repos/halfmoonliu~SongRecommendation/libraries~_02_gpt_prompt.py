"""
GPT Prompt Script

This script utilizes the OpenAI GPT-3 (text-davinci-003) engine to generate responses based on user input,
specifically for requesting music recommendations. It defines a function 'get_resp_gpt' that takes a user's
mood as input, formulates a prompt, and retrieves a GPT-3 generated response with recommended songs.
"""


import openai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


def get_resp_gpt(user_input, api_key):
    """
    Generate a prompt for GPT-3 and get a response.

    Args:
    user_input (str): The user's input to set the mood for music recommendations.
    api_key (str): The OpenAI GPT-3 API key.

    Returns:
    str or None: The generated response from GPT-3 or None in case of an error.
    """
    if not api_key:
        raise ValueError("API key not found in environment variables")

    prompt = f"I'm in the mood for music because {user_input}. Recommend 10 songs (in english only) that match my mood. Please provide the song titles first, followed by the artists' names, all together."

    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            api_key=api_key,
        )

        return response.choices[0].text

    except Exception as e:
        print(f"Error in GPT request: {e}")
        return None
