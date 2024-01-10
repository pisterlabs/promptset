import openai
import json
import os
import argparse

# Load API key, JSON file path and context via command-line arguments or environment variables
parser = argparse.ArgumentParser(description="Add description to JSON using OpenAI API key")
parser.add_argument('--key', type=str, default=os.getenv('OPENAI_API_KEY'), help="OpenAI API Key")
parser.add_argument('--file', type=str, default=os.getenv('JSON_FILE'), help="Path to JSON file")
parser.add_argument('--context', type=str, default=os.getenv('PROMPT_CONTEXT'), help="Context for the prompt")
args = parser.parse_args()

openai.api_key = args.key
json_file = args.file
prompt_context = args.context

# Function to generate description using GPT-3
def generate_description(title):
    response = openai.Completion.create(
        engine="text-davinci-003",  # Using the DaVinci model
        prompt=f"Context: {prompt_context}\nDescribe the following {title} in markdown format (only using headings) and do not use more than 180 words",
        temperature=0.5,
        max_tokens=300
    )
    description = response.choices[0].text.strip()
    return description + "\n\nAbout: " + prompt_context

# Recursive function to process the JSON structure
def process_json_structure(data):
    if isinstance(data, dict):
        for key, value in list(data.items()):  # Create a list from items
            if isinstance(value, dict) or isinstance(value, list):
                process_json_structure(value)
            if key == 'title' and 'description' not in data:  # Check if 'description' already exists
                data['description'] = generate_description(value)
    elif isinstance(data, list):
        for item in data:
            process_json_structure(item)

# Load your JSON file
with open(json_file, 'r') as f:
    data = json.load(f)

# Process the JSON structure
process_json_structure(data)

# Write the updated JSON back to a new file
new_json_file = json_file.rsplit('.', 1)[0] + '_added.json'
with open(new_json_file, 'w') as f:
    json.dump(data, f, indent=4)
