import os
import openai
import json
from dotenv import load_dotenv
import random
import time
import google.generativeai as genai
import glob

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
deepinfra_api_key = os.getenv("DEEPINFRA_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)

# Define the maximum number of retries for API calls (for one entry)
MAX_RETRIES = 5

# Define a function to construct an example from an entry
def get_example(entry):
    # Construct an example from an entry
    question = "Question:\n" + entry["question_variation"]
    answer = "Code:\n" + entry["answer_variation"]
    return f"{question}\n\n{answer}\n"

# Get the current script's directory path
script_directory = os.path.dirname(os.path.abspath(__file__))

# Find the root directory which has the name 'Experiment'
root_directory = script_directory
while os.path.basename(root_directory) != 'Experiment' and root_directory != os.path.dirname(root_directory):
    root_directory = os.path.dirname(root_directory)

# Construct paths for optimization models and experiment settings
optimization_models_path = os.path.join(root_directory, "Optimization Models")
settings_path = os.path.join(script_directory, "experiment_settings.json")

# Load experiment settings from the JSON file
with open(settings_path, "r") as settings_file:
    settings = json.load(settings_file)

# Extract relevant details from the settings
icl_number = int(''.join(filter(str.isdigit, settings["ICL"].split(":")[0])))
task_desc = settings["tasks"]
task_number = task_desc.split(":")[0][-1]
model_desc = settings["optimization_models"]
model_number = model_desc.split(":")[0][-1]
experiment_desc = settings["experiments"]
llms_model = settings["llms"].split(":")[1].strip().lower()
llms_number = settings["llms"].split(":")[0][-1]

# Check if the LLM is Codellama to change the API key and base URL
if llms_number == "3":
    # Update the client with Codellama's API base URL and DEEPINFRA_API_KEY
    client = openai.OpenAI(api_key=deepinfra_api_key, base_url="https://api.deepinfra.com/v1/openai")
    llms_model = "codellama/CodeLlama-34b-Instruct-hf"
else:
    # Use the standard OPENAI_API_KEY for other models
    client = openai.OpenAI(api_key=openai_api_key)

# Gemni Pro function to be used instead of an API call to OpenAI
def gemini_pro(prompt):
    # Assuming the prompt is a list of dictionaries with 'role' and 'content' keys
    formatted_contents = [{'text': message['content']} for message in prompt]

    # Create a model
    model = genai.GenerativeModel('gemini-pro')

    # Generate content
    response = model.generate_content(
        formatted_contents,
        generation_config=genai.types.GenerationConfig(
            temperature=0)
    )

    return response.text

# Construct folder names and input JSON filename based on the extracted details
model_folder_name = model_desc.strip().split(
    ":")[0] + ": " + model_desc.strip().split(":")[1].strip()
task_folder_name = task_desc.strip().split(
    ":")[0] + ": " + task_desc.strip().split(":")[1].strip()
base_path = os.path.join(root_directory, "Question Generation",
                         model_folder_name, task_folder_name, experiment_desc.strip())
input_json_filename = os.path.join(
    base_path, f"JSON3_reformulation_{model_number}_{task_number}.json")

# Construct system message and output filename based on task number
if task_number == "1":
    system_message = "Based on the user's description, please construct a complete, executable Python optimization model that utilizes the Gurobi solver. The output of the Python model should be: the selected items' indices as a list (called selected_items),and the objective value (called objective_value). Your response should ONLY contain the complete and executable Python model. Do NOT include any explanations, introductions, partial sections, or excerpts. The answer should strictly consist of the full model in its entirety, ready for direct execution."
    if icl_number > 0:
        system_message += "\n\nHere are some example questions and their correct codes:\n— EXAMPLES —\n\n{selected_examples}\n\n—"
    log_message = "Model {} built"
    output_filename = os.path.join(
        script_directory, f"JSON4_llm_response_{model_number}_{task_number}.json")
elif task_number == "2":
    # Use glob to find the file that matches the pattern "model{model_number}_*.py"
    model_files = glob.glob(os.path.join(optimization_models_path, f"model{model_number}_*.py"))

    if model_files:
        # Open the first file from the list
        with open(model_files[0], "r") as model_file:
            model_content = model_file.read()
    else:
        print(f"No file found for model{model_number}")
        model_content = "Error: Model file not found."

    system_message = "Please transform the provided Python optimization model based on the user's question. Your response should ONLY contain the complete, transformed, and executable Python model. Do NOT include any explanations, introductions, partial sections, or excerpts. The answer should strictly consist of the full modified model in its entirety, ready for direct execution.\n\n— OPTIMIZATION MODEL TO TRANSFORM —\n\n{model_content}\n\n—"
    if icl_number > 0:
        system_message += "\n— EXAMPLES —\n\n{selected_examples}\n\n—"
    log_message = "Model {} transformed"
    output_filename = os.path.join(
        script_directory, f"JSON4_llm_response_{model_number}_{task_number}.json")

# Load the input data from the JSON file
with open(input_json_filename, "r") as f:
    data = json.load(f)

# Process each variation in the data, construct user messages and make API calls
for i, variation in enumerate(data['variations']):
    # Changed from question_reformulation
    question = variation['question_variation']
    entry_id = variation['id']
    question_set_number = int(entry_id.split(".")[2])
    selected_examples_content = ""  # Initialize this to empty

    example_ids = {}
    if icl_number > 0:
        if settings["distribution"] == "Dist0: Out-of-distribution_ICL":
            example_pool = [e for e in data['variations'] if int(
                e['id'].split(".")[2]) != question_set_number]
        else:
            current_variation_number = int(entry_id.split(".")[3])
            example_pool = [e for e in data['variations'] if int(e['id'].split(
                ".")[2]) == question_set_number and int(e['id'].split(".")[3]) != current_variation_number]

        selected_examples = random.sample(
            example_pool, min(icl_number, len(example_pool)))
        selected_examples_content = "\n\n".join(
            [get_example(e) for e in selected_examples])
        for idx, example in enumerate(selected_examples, 1):
            example_ids[str(idx)] = example['id']

    user_messages = [
    {"role": "system", "content": system_message.format(
        model_content=model_content if task_number == "2" else "", 
        selected_examples=selected_examples_content
    )},
    {"role": "user", "content": question}
]

    # Print details before sending to the API
    print("\n--- Sending to API ---")
    print("Model:", llms_model)
    for msg in user_messages:
        print(f"  Role: {msg['role']}")
        print(f"  Content:\n    {msg['content']}\n")
    print("Entry ID:", entry_id)
    print("------------------------\n")

    # Modify the API call section
    if llms_number == "4":
        # Use the gemini_pro function instead of making an API call with the OpenAI API
        response_text = gemini_pro(user_messages)
        llm_model_response = response_text 
    else:
        # Existing API call logic
        retries = 0
        success = False
        while not success and retries < MAX_RETRIES:
            try:
                response = client.chat.completions.create(model=llms_model,
                                                        messages=user_messages,
                                                        temperature=0,
                                                        seed=1234)
                llm_model_response = response.choices[0].message.content  # Extracting content from OpenAI API response
                success = True  # if no exception, mark as success
            except openai.APIError as e:
                print(f"OpenAI API returned an API Error: {e}, retrying in 1 min...")
                retries += 1
                time.sleep(60)
            except Exception as e:  # Catching any other unexpected exceptions
                print(f"Unexpected error: {e}, retrying in 1 min...")
                retries += 1
                time.sleep(60)    

        if not success:
            print(f"Failed to process entry {entry_id} after {MAX_RETRIES} attempts.")
            continue  # skip the rest of the loop iteration

    # Process the API response and update the data
    data['variations'][i]['llm_model'] = llm_model_response

    if example_ids:
        data['variations'][i]['icl_example_ids'] = example_ids

    # Print log message
    print(log_message.format(entry_id))

# Save the updated data to the output JSON file
with open(output_filename, "w") as f:
    json.dump(data, f, indent=4)