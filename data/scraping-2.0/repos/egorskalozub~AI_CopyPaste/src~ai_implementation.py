import openai
import os
import pyperclip
import logging
from settings_manager import get_api_key, load_settings


def ask_openai(question, user_temperature, user_max_tokens, use_history):
    print(question)
    print(user_temperature)
    print(user_max_tokens)
    print(use_history)
    # Load the settings from the settings.json file
    openai.api_key = f"{get_api_key()}"
    settings = load_settings()
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=question.strip(),
        temperature=user_temperature,
        max_tokens=user_max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # Get the first choice from the response object and store it in variable 'answer'
    answer = response.choices[0].text.strip()

    # Copy the answer to clipboard using pyperclip module
    pyperclip.copy(answer)

    # Create a separator line for ease of readability
    break_line = "=" * 40

    # Print the separator line to console
    print(response)
    print(answer)
    # Log the question, response, and answer using python logging module
    logging.info(f"Question: {question}")
    logging.info(f"Response: {response}")
    logging.info(f"Answer: {answer}")

    # Return the answer so that it can be used by other parts of the program
    return answer
