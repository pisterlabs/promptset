import os
import json
import hashlib
import time
from openai import OpenAI
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

# Function to compute SHA256 hash of a given string
def compute_sha256(input_string):
    return hashlib.sha256(input_string.encode()).hexdigest()

def reformulate_question(question, reformulated_hashes, task_num):
    unique_reformulation = False
    reformulated_question = ""
    reformulation_hash = ""
    max_retries = 5
    retries = 0
    
    # Change system prompt based on task number
    system_prompt = "Please reformulate the question without changing the meaning. Don't reformulate the word index"
    if task_num == 1:
        system_prompt = "Please reformulate the description of this optimization model without changing the meaning. Don't reformulate the word index"
    
    while not unique_reformulation and retries < max_retries:
        success = False
        api_retries = 0  # Initialize counter for API retries
        MAX_API_RETRIES = 3  # Define the maximum number of retries for API calls (for one entry)
        while not success and api_retries < MAX_API_RETRIES:
            try:
                response = client.chat.completions.create(model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                temperature= 0.7,
                )
                reformulated_question = response.choices[0].message.content
                success = True  # if no exception, mark as success
            except openai.OpenAIError as e:
                print(f"Error {e} occurred, retrying in 1 min...")
                retries += 1
                time.sleep(60)
            except Exception as e:  # Catching any other unexpected exceptions
                print(f"Unexpected error: {e}, retrying in 1 min...")
                retries += 1
                time.sleep(60)

        if not success:
            print(f"Warning: Couldn't call OpenAI API for '{question}' after {MAX_API_RETRIES} attempts.")
            return reformulated_question, reformulation_hash  # early exit in case of persistent API failure

        reformulation_hash = compute_sha256(reformulated_question)
        if reformulation_hash not in reformulated_hashes:
            reformulated_hashes.add(reformulation_hash)
            unique_reformulation = True
        else:
            print("Generated question is regenerated due to equality with previous question.")
            retries += 1

    if not unique_reformulation:
        print(f"Warning: Couldn't get a unique reformulation for '{question}' after {max_retries} attempts.")

    return reformulated_question, reformulation_hash

# Get the directory of the currently executing script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to experiment_settings.json
settings_path = os.path.join(script_directory, "experiment_settings.json")

# Load the settings from the experiment_settings.json file
with open(settings_path, "r") as settings_file:
    settings = json.load(settings_file)

# Construct the model and task numbers
model_number = settings["optimization_models"].split("Model")[1].split(":")[0].strip()
task_number = settings["tasks"].split("Task")[1].split(":")[0].strip()
reformulations_per_variation = int(settings["reformulations_per_variation"])

# Construct the full path to input and output JSON files
input_file_path = os.path.join(script_directory, f'JSON2_solved_variations_{model_number}_{task_number}.json')
output_file_path = os.path.join(script_directory, f'JSON3_reformulation_{model_number}_{task_number}.json')

# Load the input JSON file
with open(input_file_path, 'r') as infile:
    data = json.load(infile)

# Initialize the output data structure and a set to store the hashes of reformulated questions
output_data = {"variations": []}
reformulated_hashes = set()

# Loop through each variation
for variation in data["variations"]:
    # Generate reformulated questions as per the number specified in the settings
    for i in range(1, reformulations_per_variation + 1):
        # Print the console log message
        print(f"Reformulating question {variation['id']}.{i}")
        new_variation = variation.copy()
        new_id = f"{variation['id']}.{i}"
        new_variation["id"] = new_id
        new_question, new_hash = reformulate_question(variation['question_variation'], reformulated_hashes, int(task_number))
        new_variation["question_reformulation"] = new_question
        new_variation["reformulation_hash"] = new_hash
        output_data["variations"].append(new_variation)

# Save to a new JSON file
with open(output_file_path, 'w') as outfile:
    json.dump(output_data, outfile, indent=4)
