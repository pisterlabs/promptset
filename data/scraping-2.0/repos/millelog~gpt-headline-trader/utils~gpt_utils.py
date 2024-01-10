import openai
from typing import Any, Dict, List
from typing import Dict, Any
from config import OPENAI_API_KEY
import time

openai.api_key = OPENAI_API_KEY
model = "gpt-3.5-turbo" # You can replace this with "gpt-4" if available and if you want to use it

def generate_prompt(headline: str, company_name: str) -> List[Dict[str, str]]:
    """
    Generates prompt for GPT-3.5-turbo or GPT-4.

    Parameters
    ----------
    headline : str
        The news headline.
    company_name : str
        The name of the company.
    term : str
        The term of interest.

    Returns
    -------
    str
        The generated prompt for GPT-3.5-turbo or GPT-4.
    """
    prompt_template = [
        {
            "role": "system",
            "content": "You are a financial expert with stock recommendation experience. Answer “YES” if good news, “NO” if bad news, or “UNKNOWN” if uncertain in the first line. Then elaborate with one short and concise sentence on the next line."
        },
        {
            "role": "user",
            "content": f"Is this headline good or bad for the stock price of {company_name} in the short term? Headline: {headline}"
        }
    ]
    return prompt_template


# Define the rate limit parameters
MAX_CALLS = 30
PERIOD = 60  # In seconds
LAST_CALL = time.time()
RETRY_DELAY = 5  # Delay in seconds before retrying after a rate limit error
MAX_RETRIES = 3  # Maximum number of retries


def get_gpt3_response(prompt: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Gets GPT-3.5-turbo or GPT-4's response to a prompt.

    Parameters
    ----------
    prompt : List[Dict[str, str]]
        The prompt for GPT-3.5-turbo or GPT-4.

    Returns
    -------
    Dict[str, Any]
        The response from GPT-3.5-turbo or GPT-4.
    """
    global LAST_CALL
    retries = 0

    while retries < MAX_RETRIES:
        try:
            time_since_last_call = time.time() - LAST_CALL
            if time_since_last_call < 1/MAX_CALLS*PERIOD:
                time.sleep(1/MAX_CALLS*PERIOD - time_since_last_call)

            response = openai.ChatCompletion.create(
                model=model,
                messages=prompt
            )

            LAST_CALL = time.time()
            return response['choices'][0]['message']['content'].strip().replace('\n', ' ')

        except openai.error.RateLimitError:
            print("RateLimitError occurred. Retrying after a delay...")
            retries += 1
            time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            return None




def process_gpt3_response(message_content: Dict[str, Any]) -> float:
    """
    Processes GPT-3.5-turbo or GPT-4's response and converts it to a score.

    Parameters
    ----------
    response : Dict[str, Any]
        The response from GPT-3.5-turbo or GPT-4.

    Returns
    -------
    float
        The score derived from GPT-3.5-turbo or GPT-4's response.
    """
    if not message_content:
        return None
    # Check the first word of the response
    first_word = message_content.split(' ', 1)[0].upper()

    # Map first word to score based on the recommendation
    if "YES" in first_word or "GOOD" in first_word:
        return 1.0
    elif "UNKNOWN" in first_word or "UNCERTAIN" in first_word:
        return 0.0
    elif "NO" in  first_word or "BAD" in first_word:
        return -1.0
    else:
        # In case there's an unexpected response, it could be handled here.
        # We can either raise an error or log the unexpected response and return a default score.
        print(f"Unexpected GPT-3.5-turbo response: {message_content}")
        return 0.0  # default score

