#!/usr/bin/env python3

import openai
import argparse
import textwrap
import os

# Set your API key here
openai.api_key = 'your-api-key'

def interact(max_width):
    print("Welcome to heat-gpt")
    use_git_repo = input("Would you like to store your templates in a local project folder? (Y/N) ")

    if use_git_repo.lower() == "y":
        repo_path = input("Please enter the local path to your project folder: ")
        save_path = repo_path
    else:
        repo_path = None
        save_path = os.getcwd()  # Save in the current working directory
        print("Templates will be stored in the current folder.")

    # Define the prompt
    prompt_prefix = '''
    The following question denoted by "QUESTION: " is to be interpreted as a task for an OpenStack heat template,
    and the desired output must be in the form of a heat template.

    For example, if the question was "Create a VM with 2 CPUs and 4GB of RAM", the output might be:

    heat_template_version: wallaby

    description: create an instance

    resources:
      my_vm:
        type: OS::Nova::Server
        properties:
          flavor: m1.small
          image: cirros
          networks:
          - network: private


    Ensure all responses are complete and functional templates with the appropriate heat headers and sections.
    Omit all but the required elements of the template.

    Now, QUESTION:
    '''

    while True:
        try:
            print('Please enter your prompt')
            message = input('> ')
            full_prompt = f'{prompt_prefix.strip()} {message}'

            while True:
                response = openai.Completion.create(
                  engine="text-davinci-003",
                  prompt=full_prompt,
                  max_tokens=2000  # Increase the max_tokens limit
                )

                response_text = response.choices[0].text.strip()

                # Check if the response is likely to be YAML
                if ':' in response_text and '\n' in response_text:
                    # Don't wrap the text
                    print(response_text)
                else:
                    # Wrap the text as before
                    wrapper = textwrap.TextWrapper(width=max_width)
                    formatted_response = wrapper.fill(text=response_text)
                    print(formatted_response)

                # Ask the user if they accept the suggestion
                decision = input('Do you accept this suggestion? (Y/N/R) ')
                if decision.lower() == 'y':
                    # Ask the user for a name for the heat template
                    filename = input('Please enter a name for the heat template: ')

                    # Remove any existing file extension from filename
                    filename = os.path.splitext(filename)[0]

                    # Add .yaml extension
                    filename += '.yaml'

                    # Write the response to a file
                    file_path = os.path.join(save_path, filename) if repo_path else filename
                    with open(file_path, 'w') as f:
                        f.write(response_text)
                    print(f'Template saved as {file_path}')

                    break

                elif decision.lower() == 'n':
                    # Go back to the main prompt
                    break
                elif decision.lower() == 'r':
                    # Regenerate the answer
                    continue
                else:
                    print("Invalid response. Please enter Y, N, or R.")

        except KeyboardInterrupt:
            print('\nExiting...')
            break
        except EOFError:
            print('\nExiting...')
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Interact with ChatGPT.')
    parser.add_argument('--max-width', type=int, default=80,
                        help='max width of the output in characters')

    args = parser.parse_args()
    interact(args.max_width)
