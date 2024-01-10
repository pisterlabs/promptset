import nltk
import os
from openai import OpenAI

api_key = 'API_KEY'

client = OpenAI()

def summarize_text(input_text):
    # Set the model and token limit
    model = "ft:gpt-3.5-turbo-1106:personal::8UJ8Or8k" #! need to change this to the current finetuned model
    max_tokens = 4000
    text_to_gpt = f"Make an abstract for the following paper: \n{input_text}"
    text_to_gpt = truncate_text(text_to_gpt, max_tokens)
    # Call the OpenAI API to generate text
    response = client.chat.completions.create(model=model,
                                              messages=[ #! maybe change?
                                                  {"role": "user", "content": text_to_gpt}]
                                              )

    # Extract the generated text from the response
    generated_text = response.choices[0].message.content
    return generated_text


def truncate_text(text, max_tokens):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Truncate to the specified number of tokens
    truncated_tokens = tokens[:max_tokens]

    # Join the tokens back into a truncated text
    truncated_text = ' '.join(truncated_tokens)

    return truncated_text


# Define the file paths
input_directory = '../data/short/shrink_paper'
output_directory = '../data/short/fine_results'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# List all files in the input directory
input_files = os.listdir(input_directory)

# Process each file
gpt_files = 0
for file_name in input_files:
    input_file_path = os.path.join(input_directory, file_name)
    output_file_path = os.path.join(output_directory, file_name)

    with open(input_file_path, 'r') as file:
        text = file.read()
    # Generate the summary
    summary = summarize_text(text)

    # Save the summary in the output directory with the same filename
    with open(output_file_path, 'w', errors="ignore") as output_file:
        output_file.write(summary)

    gpt_files += 1
    if gpt_files >= 1: #adjust as needed
        break

print(f"Summarized and saved {gpt_files} files to {output_file_path}")