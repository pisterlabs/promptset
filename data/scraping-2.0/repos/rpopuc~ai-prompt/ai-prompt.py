#!/usr/bin/python3
import openai
import re
import os
from colorama import Fore, Style, init
from enquiries import choose
from pathlib import Path

def get_multiline_input(prompt):
    lines = []
    print(prompt)
    while True:
        try:
            line = input()
        except EOFError:
            break
        lines.append(line.strip())
    return ('\n'.join(lines)).strip()

def replace_words(text):
    replaced_words = []

    # Find all words within braces "{word}"
    found_words = re.findall(r'\{(\w+)\}', text)

    # Prompt the user for content for each found word
    print(f'\n{Fore.YELLOW}Enter values for macros. Press CTRL+D to complete entries.{Style.RESET_ALL}')
    for word in found_words:
        content = get_multiline_input(f'\n{Fore.GREEN}{word.capitalize()}{Style.RESET_ALL}:')
        replaced_words.append((word, content))

    # Replace the words in the text with the user-provided values
    for word, content in replaced_words:
        text = text.replace(f'{{{word}}}', content)

    return text


def main():
    # Directory name of the subdirectory within the user's home folder
    subdirectory_name = 'prompt-templates'

    # Get the path to the user's home folder
    home_path = Path.home()

    # Build the full path to the subdirectory within the user's home folder
    templates_directory = home_path / subdirectory_name

    # Check if the subdirectory exists
    if not templates_directory.exists() or not templates_directory.is_dir():
        print(f'The subdirectory "{subdirectory_name}" does not exist in the user\'s home folder.')
        return

    # List all files in the subdirectory
    template_files = os.listdir(templates_directory)

    # Check if there are any template files
    if not template_files:
        print('No template files found.')
        return

    # Display the available template files
    template_choices = [f'{i+1}. {template_file}' for i, template_file in enumerate(template_files)]
    template_choices.insert(0, '0. Exit')
    template_choice = choose('Choose the prompt template:', template_choices)

    try:
        # Convert the user's choice to an integer
        template_choice = template_choices.index(template_choice)

        if template_choice == 0:
            return

        # Check if the choice is within the valid range
        if template_choice < 0 or template_choice >= len(template_choices):
            print('Invalid template file choice.')
            return

        # Get the selected template file
        selected_template_file = template_files[template_choice - 1]

        # Build the full path to the selected template file
        template_path = os.path.join(templates_directory, selected_template_file)

        with open(template_path, 'r') as file:
            text = file.read()

            # Call the function to replace the words in the text
            prompt = replace_words(text)

            print(f'\n{Fore.GREEN}Prompt:{Style.RESET_ALL}')

            print(prompt)

            print(f'\n{Fore.GREEN}Resposta...{Style.RESET_ALL}')

            openai.api_key = os.getenv('OPENAI_API_KEY')
            complete = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                temperature=0.7,
                max_tokens=1000
            )
            response = complete.choices[0].text.strip()
            print(response)

    except ValueError:
        print('Invalid template file choice.')


if __name__ == '__main__':
    main()