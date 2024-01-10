#!/usr/bin/python3
import openai
import os
import time

def print_with_typing_effect(text, delay=0.005):
    """Print text with a typing effect."""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()  # Newline at the end

def insert_into_latex(gpt_response, section_title):
    """Insert the GPT-4 response into a LaTeX file."""
    latex_file_path = '../results/notebook/notebook.tex'
    
    with open(latex_file_path, 'r') as latex_file:
        content = latex_file.readlines()
        
    end_document_index = None
    for idx, line in enumerate(content):
        if '\\end{document}' in line:
            end_document_index = idx
            break
    
    # If the \end{document} tag is found, insert the new content before it
    if end_document_index is not None:
        content.insert(end_document_index, '\n\\section{' + section_title + '}\n')
        content.insert(end_document_index + 1, gpt_response + '\n')
        with open(latex_file_path, 'w') as latex_file:
            latex_file.writelines(content)
    else:
        print("Couldn't find \\end{document} in the LaTeX file.")

def ask_gpt4():
    """Get a user prompt, ask GPT-4, and append the response to a LaTeX file."""
    # Fetch the OpenAI API key from environment variable
    openai_key = os.environ.get('OPENAI_KEY')
    if not openai_key:
        raise ValueError("Please set your OpenAI API key as the 'OPENAI_KEY' environment variable.")
    
    openai.api_key = openai_key
    
    # Get section title
    section_title = input("Enter the section title for LaTeX: ")

    # Get user prompt for GPT-4
    prompt = input("Enter your prompt for GPT-4: ")

    # Send the prompt to GPT-4
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides definitions and descriptions."},
            {"role": "user", "content": prompt}
        ]
    )

    gpt_response = response.choices[0].message["content"]

    print("\nGPT-4 Response:")
    print_with_typing_effect(gpt_response)

    # Insert the response into the LaTeX file
    insert_into_latex(gpt_response, section_title)

if __name__ == "__main__":
    ask_gpt4()

