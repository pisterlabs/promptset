# Auto-Archive: Clericus Module
# Version: v0.0.3-dev
# Path: `\core\clericus.py`
# Updated: 09-24-2023

# ⚠️ ACTIVE DEVELOPMENT ⚠️ #

import openai
import json
from typing import Union
from .clericus_modules.web_search import perform_web_search
from .clericus_modules.openai_module import generate_openai_response
from core.logger import clericus_logger, handle_generic_error
from core.config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY


def generate_response(user_input: str, session_id: str) -> Union[str, dict]:
    """
    Generate a response based on user input and session ID.
    If 'web_search' is in the response, perform a web search.
    """
    try:
        response = generate_openai_response(user_input, session_id)
        if 'web_search' in response:
            search_results = perform_web_search(user_input)
            return json.dumps(search_results)  # Convert object to JSON string
        return response  # Assuming this is already a string
    except Exception as e:
        error_response = handle_generic_error(e, clericus_logger)
        return json.dumps(error_response)  # Convert object to JSON string
# Function to format the response text into more readable chunks like paragraphs or lists


def format_response_text(response_text: str) -> str:
    # Keywords or phrases that could indicate a list or new paragraph
    list_indicators = ["- ", "* ", "1. ", "2. ", "3. "]
    paragraph_indicators = ["\n\n"]

    # Initialize variables
    formatted_text = ""
    current_line = ""

    # Tokenize the text by spaces
    tokens = response_text.split(" ")

    for token in tokens:
        current_line += token + " "

        # Check for list or new paragraph indicators
        if any(indicator in current_line for indicator in list_indicators):
            formatted_text += "\n- " + current_line.strip()
            current_line = ""
        elif any(indicator in current_line for indicator in paragraph_indicators):
            formatted_text += "\n\n" + current_line.strip()
            current_line = ""

    # Append any remaining text
    formatted_text += current_line.strip()

    return formatted_text
