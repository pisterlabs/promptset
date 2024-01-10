import openai
import os
import asyncio
import json
import aiohttp
import tiktoken
from util import num_tokens_from_string
from dotenv import load_dotenv

load_dotenv()

MODEL = "gpt-3.5-turbo"

def generate_system_prompt():
    return """
Translate the given page titles to English.
Input: JSON array of page titles.
Output: JSON dictionary of original titles & translated titles.
"""

# global variables for log output
translation_start_count = 1
translation_end_count = 1

# Set OpenAI API Key
openai.api_key = os.environ.get("OPENAI_API_KEY")

async def fetch_title_translation(title_array_str, max_retries=3):
    """
    Fetches the translation of a title from OpenAI API

    Args:
        title_array_str (str): The array of titles to be translated
        max_retries (int): The maximum number of retries in case of failure

    Returns:
        (dict): A dictionary of original titles & translated titles
        
    """
    temperature = 0

    for attempt in range(max_retries):
        try:
            global translation_start_count
            print("(#" + str(translation_start_count) + ") Starting Title Chunk Translation")
            translation_start_count += 1

            response = await openai.ChatCompletion.acreate(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": generate_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": title_array_str
                    }
                ],
                temperature=temperature,
                max_tokens=3000
            )

            # For each returned translation, add to the translations dictionary
            response_content = response['choices'][0]['message']['content']
            translated_titles = json.loads(response_content)

            global translation_end_count
            print("(#" + str(translation_end_count) + ") Finished Title Chunk Translation")
            translation_end_count += 1

            return translated_titles
        except Exception as e:
            print(f"Error in fetch_title_translation, attempt #{attempt + 1}")
            print(e)
            if attempt == max_retries - 1:  # If this was the last attempt
                # Create dictionary of {original_titles: original_titles}
                return {title: title for title in json.loads(title_array_str)}
            else:
                temperature += 0.5
                print("Retrying...")
