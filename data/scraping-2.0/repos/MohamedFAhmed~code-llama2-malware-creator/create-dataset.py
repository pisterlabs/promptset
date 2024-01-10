import os
import time
import requests
import json
import openai

# Your OpenAI API key
openai.api_key = os.getenv('OPENAI_KEY')
if openai.api_key is None:
    raise ValueError("API key not found as environment variable")


# The OpenAI API URL for GPT-4
API_URL = 'https://api.openai.com/v1/engines/gpt-4/completions'

# Function to call the ChatGPT API
def call_chatgpt_api(source_code):
   # try:
    response = openai.chat.completions.create(
         model="gpt-4-1106-preview",
        response_format={ "type": "json_object" },
      #  prompt=f"I'm creating a training dataset that consists of (1) instruction, (2) task, (3) inputs, and (4) response. "
      #          f"The response should be a source code similar to the one I will feed you below. "
      #          f"I want you to create the instructions, task, and possible inputs based on the source code below, "
      #          f"which will be the response the user is supposed to get.\n\n{source_code}",
         messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": f"I'm creating a training dataset that consists of (1) instruction, (2) task, (3) inputs, and (4) response. "
                f"The response should be a source code similar to the one I will feed you below. "
                f"I want you to create the instructions, task, and possible inputs based on the source code below, "
                f"which will be the response the user is supposed to get.\n\n{source_code}"}
         ]
       # max_tokens=1024
    )
    return response
   # except openai.error.OpenAIError as e:  # Adjusted to the new error handling
   #     print(f"An error occurred: {e}")

def format_nested_items(d, indent=0):
    """Recursively format nested dictionaries and lists into a string with indentation."""
    lines = []  # Collect all the lines in this list
    if isinstance(d, dict):
        for key, value in d.items():
            if isinstance(value, (dict, list)):
                lines.append(' ' * indent + f"{key}:")
                lines.append(format_nested_items(value, indent + 2))
            else:
                lines.append(' ' * indent + f"{key}: {value}")
    elif isinstance(d, list):
        for item in d:
            if isinstance(item, (dict, list)):
                lines.append(format_nested_items(item, indent + 2))
            else:
                lines.append(' ' * indent + f"- {item}")
    return '\n'.join(lines)

def format_dataset_entry(response_json, source_code):
    # Parse the JSON string into a dictionary
    api_response = json.loads(response_json)
    
    # Extract the instruction and format it
    instruction = api_response.get('instruction', '').strip()

    # Check if task is a list and format accordingly
    task = api_response.get('task', '')
    if isinstance(task, list):
        task_formatted = "\n".join(f"  - {item.strip()}" for item in task)
    else:
        task_formatted = task.strip()
    
    # Extract and format inputs
    inputs = api_response.get('inputs', {})
    inputs_formatted = format_nested_items(inputs, 2)
    
    # The source code is included as provided
    response_code = source_code.strip()

    # Formatting the dataset entry
    return f"""
===BEGIN DATASET===
INSTRUCTION: {instruction}
TASK:
{task_formatted}
INPUTS:
{inputs_formatted}
RESPONSE: 
{response_code}
===END DATASET===
"""

def read_source_code(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def write_dataset_entry(dataset_entry, output_file):
    with open(output_file, 'a') as file:
        file.write(dataset_entry + '\n')

def main(source_folder, output_file):
    for file_name in os.listdir(source_folder):
        if file_name.endswith(('.py', '.js', '.java', '.c', '.cpp', '.pl')):  # Add other file extensions as needed
            file_path = os.path.join(source_folder, file_name)
            source_code = read_source_code(file_path)
            attempt = 0
            max_attempts = 5  # Set a max number of attempts to avoid infinite loops
            while attempt < max_attempts:
                try:
                    response = call_chatgpt_api(source_code)
                    formatted_entry = format_dataset_entry(response.choices[0].message.content, source_code)
                    write_dataset_entry(formatted_entry, output_file)
                    break  # Break out of the loop if successful
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:
                        wait_time = int(e.response.headers.get('Retry-After', 60))  # Use the Retry-After header or default to 60 seconds
                        print(f"Rate limit exceeded. Waiting for {wait_time} seconds before retrying...")
                        time.sleep(wait_time)
                        attempt += 1  # Increment the attempt counter
                    else:
                        raise  # Re-raise the exception if it's not a 429 error
                except requests.exceptions.RequestException as e:
                    print(f"An error occurred: {e}")
                    break  # Break the loop if there's a non-recoverable error

# Set your source folder and the output dataset file path
source_folder = 'training/ddos'
output_file = 'ddos_dataset.txt'

if __name__ == '__main__':
    main(source_folder, output_file)
