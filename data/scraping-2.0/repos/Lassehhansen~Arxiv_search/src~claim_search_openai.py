import json
import openai
import time
from tqdm import tqdm
import pandas as pd
from typing import List, Tuple

def create_keyword_pattern(keywords):
    """
    Create a regex pattern for keyword matching.
    """
    pattern = r'(?:(?<=\W)|(?<=^))(' + '|'.join(map(re.escape, keywords)) + r')(?=\W|$)'
    return re.compile(pattern, re.IGNORECASE)

def get_context_windows(text, keywords, window_size):
    """
    Extract context windows around keywords in the text, ensuring that overlapping
    windows are merged into one.
    """
    context_windows = []
    keyword_pattern = create_keyword_pattern(keywords)

    matches = list(keyword_pattern.finditer(text))
    merged_matches = []

    i = 0
    while i < len(matches):
        current_match = matches[i]
        start_pos = current_match.start()
        end_pos = current_match.end()

        # Check if the next matches are within the window size
        while i + 1 < len(matches) and matches[i + 1].start() - end_pos <= window_size * 2:
            i += 1
            end_pos = matches[i].end()

        merged_matches.append((start_pos, end_pos))
        i += 1

    for start_pos, end_pos in merged_matches:
        words = text.split()
        start_word_pos = len(text[:start_pos].split()) - 1
        end_word_pos = len(text[:end_pos].split())
        context_words = words[max(0, start_word_pos - window_size):min(len(words), end_word_pos + window_size)]
        context_str = ' '.join(context_words)
        context_windows.append(context_str)

    return context_windows

def process_with_gpt(context_window, model, system_prompt, openai_api_key, delay=1.0):
    """
    Processes a context window with the GPT model to extract relevant information based on the given prompt.

    Parameters:
    - context_window (str): The context window text to be processed.
    - model (str): The OpenAI GPT model to be used.
    - system_prompt (str): The prompt to be used for querying the GPT model.
    - openai_api_key (str): The API key for OpenAI.
    - delay (float): Delay between API calls in seconds to manage rate limits.

    Returns:
    - str: The GPT response as a string.
    """
    openai.api_key = openai_api_key
    retry_delay = 1  # Initial delay in seconds for retries
    max_retries = 5  # Maximum number of retries

    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context_window}
                ]
            )
            return response.choices[0].message.content
        except openai.error.RateLimitError as e:
            print(f"Rate limit reached, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
        except Exception as e:
            print(f"Failed processing context window with error: {e}")
            return f"Error: {str(e)}"

        time.sleep(delay)  # Delay between API calls

    return "Error: Rate limit retries exceeded"
