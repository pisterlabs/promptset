# This file will use the GPT-4 API to generate a bibtex file from a plain text file

import os
import openai

openai.api_type = "azure"
openai.api_base = "https://nul-staff-openai.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")
# Set the API key
    
# Set the temperature
temperature = 0.0
# Set the max tokens
max_tokens = 500
# Set the top p
top_p = 1
# Set the frequency penalty
frequency_penalty = 0.5
# Set the presence penalty
presence_penalty = 0.0
# get content of file

def get_input_content(input_path):
    with open(input_path, 'r') as f:
        content = f.read()
    chunks = content.split('\n')
    print(len(chunks))
    return chunks, input_path


def call_gpt(content_chunks):
    responses = []
    for chunk in content_chunks:
        try:
            print(f"\n\n\nInput: {chunk}")
            completion = openai.ChatCompletion.create(
                engine="nul-general-gpt35",
                messages=[{"role": "user", "content": str(chunk)},
                          {"role": "system", "content": "You are an AI that converts plaintext citations to biblatex. Only respond with code in plain text."}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=["\n}"],
            )
            response = completion.choices[0].message.content + "\n}"
            print(response)
            responses.append(response)
            print("\n\n")
            print(responses)
        except Exception as e:
            print(f"Error occurred: {e}")
            continue

    return responses

# Save responses to a bib file
def write_bib_file(responses, input_file_path):
    output_file_path = os.path.splitext(input_file_path)[0] + '.bib'
    with open(output_file_path, 'w') as f:
        for response in responses:
            f.write(response)
            f.write('\n')
    print(f"Responses saved to {output_file_path}")

import os

def gpt4bibtex():
    for root, dirs, files in os.walk(os.getcwd() + '/production'):
        for file in files:
            if file.endswith('.txt'):
                input_file_path = os.path.join(root, file)
                chunks, input_path = get_input_content(input_file_path)
                responses = call_gpt(chunks)
                write_bib_file(responses, input_file_path)
                write_api_log(responses, input_file_path)  # Write responses to API log

def write_api_log(responses, input_file_path):
    log_file_path = os.path.splitext(input_file_path)[0] + '_api.log'
    with open(log_file_path, 'w') as f:
        for response in responses:
            f.write(response)
            f.write('\n')
    print(f"API log saved to {log_file_path}")


# Call gpt4bibtex() for every .txt file within the production directory and its subdirectories

gpt4bibtex()
