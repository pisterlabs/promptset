import openai
import os
from dotenv import load_dotenv, find_dotenv
import pyperclip
import argparse
import json
import glob
import re

# Load API Key
_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv('OPENAI_API_KEY')

def validate_translation(german_text, spanish_text):
    """ Validate the corrected translation using ChatGPT in JSON mode. """
    prompt_text = (
        "Bitte bewerte die folgende Übersetzung auf Genauigkeit und Natürlichkeit. "
        "Antworte in einem strukturierten JSON-Format mit nur zwei Feldern: 'valid' und 'feedback'. "
        "Das Feld 'valid' soll nur die Werte 'true' oder 'false' enthalten. 'true' bedeutet, dass die Übersetzung "
        "sehr wahrscheinlich korrekt ist und keine wahrscheinlichen Fehler gefunden wurden. 'false' bedeutet, dass mögliche Fehler gefunden wurden.\n\n"
        f"Original: {german_text}\nÜbersetzung: {spanish_text}"
    )

    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": ""}
        ],
        temperature=0
    )

    # Write the raw response to a file for analysis
    with open("response_output.txt", "w", encoding="utf-8") as file:
        file.write(str(response))

    # Assuming the model follows the instructions, parse the response accordingly
    try:
        # Extract the content from the response
        response_content = response['choices'][0]['message']['content']
        # Parse the JSON-formatted string
        response_data = json.loads(response_content)
        validation_result = response_data['valid']
        error_message = response_data.get('feedback', 'No feedback provided.')
    except (KeyError, TypeError, json.JSONDecodeError):
        validation_result = False
        error_message = "Could not parse the response properly."

    return validation_result, error_message


def create_next_example_folder(german_text, spanish_text):
    existing_folders = glob.glob("example*/")
    highest_number = 0
    for folder in existing_folders:
        # Extract the number part of the folder name
        number_part = folder[len("example"):-1]  # Remove 'example' and the trailing slash
        if number_part.isdigit():
            highest_number = max(highest_number, int(number_part))
    
    new_folder = f"example{highest_number + 1}"
    os.makedirs(new_folder, exist_ok=True)
    
    with open(os.path.join(new_folder, "german.txt"), 'w', encoding='utf-8') as file:
        file.write(german_text)
    with open(os.path.join(new_folder, "spanish.txt"), 'w', encoding='utf-8') as file:
        file.write(spanish_text)

    return new_folder




# ... [Rest of your script] ...
def update_examples(german_text, spanish_text, force_update=False):
    validation_result, error_message = validate_translation(german_text, spanish_text)
    
    if validation_result or force_update:
        # If the translation is valid or force update is specified, create a new folder and add texts
        new_folder = create_next_example_folder(german_text, spanish_text)
        return f"Texts added to new examples folder: {new_folder}."
    else:
        # If the translation is not valid and not a force update, do not create a new folder
        return f"Possible errors found: {error_message}"


def main(force_update_arg):
    # Read the new text and corrected translation
    with open("new text/german.txt", 'r', encoding='utf-8') as file:
        german_text = file.read()
    with open("correction/corrected.txt", 'r', encoding='utf-8') as file:
        spanish_text = file.read()

    # Check if force update argument is provided
    if force_update_arg:
        # Ask the user if they want to force update the examples
        force_update = input("Do you want to force update the examples? (y/n): ").strip().lower() == 'y'
    else:
        force_update = False

    # Update examples or report errors
    result_message = update_examples(german_text, spanish_text, force_update=force_update)
    print(result_message)

if __name__ == "__main__":
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description='Update translation examples.')
    parser.add_argument('--force-update', action='store_true', help='Force update the examples after validation')
    args = parser.parse_args()

    main(args.force_update)
