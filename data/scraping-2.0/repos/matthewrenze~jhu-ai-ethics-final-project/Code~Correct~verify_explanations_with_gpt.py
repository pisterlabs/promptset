# Import libraries
import os
import re
import csv
from datetime import datetime
import openai

# Set the GPT model
model_name = "gpt-4"
treatment = "corrected-2"
task = "verify"

# Set the file paths
root_folder_path = "C:/Users/Matthew/Dropbox/Personal/School/JHU/Ethics/Project"
system_prompt_file_path = root_folder_path + "/Data/Prompts/verify-explanations-system-prompt.txt"
template_file_path = root_folder_path + "/Data/Templates/verify-explanations-template.txt"
example_explanations_folder_path = root_folder_path + "/Data/Examples/Explanations"
example_verifications_folder_path = root_folder_path + "/Data/Examples/Verifications"
explanations_folder_path = root_folder_path + f"/Data/Explanations/{model_name}-{treatment}"
verifications_folder_path = root_folder_path + f"/Data/Verifications/{model_name}-{treatment}"
log_file_name = f"{model_name}-{treatment}-{task}.csv"
log_folder_path = root_folder_path + f"/Data/Logs"
log_file_path = log_folder_path + "/" + log_file_name

# Specify the number of examples to use (0-3)
number_of_examples = 3

# Set API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Display the initial status
print("Verifying all explanations...")

# Create output directory if it doesn't exist
if not os.path.exists(verifications_folder_path):
    os.makedirs(verifications_folder_path)

# Create the log folder if it does not exist
if not os.path.exists(os.path.dirname(log_file_path)):
    os.makedirs(os.path.dirname(log_file_path))

# Create a CSV header for the log file
log_header = [
    "Date-Time",
    "Model",
    "Treatment",
    "Task",
    "Prompt Tokens",
    "Completion Tokens",
    "Total Tokens",
    "Start Date-Time",
    "End Date-Time",
    "Duration (s)",
    "Prompt",
    "Response",
    "Error Message"]

# Write the header to the log file as append
with open(log_file_path, "a", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(log_header)

# Read the prompt file
with open(system_prompt_file_path, "r") as prompt_file:
    system_prompt = prompt_file.read()

# Read the template file
with open(template_file_path, "r") as template_file:
    template = template_file.read()

# Create replacements values for placeholders in the system prompt
replacements = {"template": template}

# Replace the placeholders in the system prompt
system_prompt = system_prompt.format(**replacements)

# Read the example explanation files
example_explanations = []
for file_name in os.listdir(example_explanations_folder_path):
    file_path = example_explanations_folder_path + "/" + file_name
    with open(file_path, "r") as file:
        example_explanations.append(file.read())

# Read the example verification files
example_verifications = []
for file_name in os.listdir(example_verifications_folder_path):
    file_path = example_verifications_folder_path + "/" + file_name
    with open(file_path, "r") as file:
        example_verifications.append(file.read())

# Get all files in the source folder
explanation_file_names = os.listdir(explanations_folder_path)

# Verify each explanation file
for explanation_file_name in explanation_file_names:

    # Display a status update
    print(f"Verifying {explanation_file_name}...")

    # Get the start time
    start_time = datetime.now()

    # Get the case ID from the first part of the file name
    case_id = int(explanation_file_name.split("-")[0])

    # Get the path to the explanation file
    explain_file_path = explanations_folder_path + "/" + explanation_file_name

    # Read the letter of explanation file
    with open(explain_file_path, "r") as explanation_file:
        explanation = explanation_file.read()

    # Create the messages
    messages = []

    # Add the system message
    system_message = {"role": "system", "content": f"{system_prompt}"}
    messages.append(system_message)

    # Add the example explanations and verifications
    for i in range(number_of_examples):
        example_explanation = {"role": "user", "content": f"{example_explanations[i]}"}
        example_verification = {"role": "assistant", "content": f"{example_verifications[i]}"}
        messages.append(example_explanation)
        messages.append(example_verification)

    # Add the user prompt
    user_message = {"role": "user", "content": f"{explanation}"}
    messages.append(user_message)

    # Set the GPT model hyper-parameters
    model_engine = "gpt-4" if model_name == "gpt-4" else "gpt-3.5-turbo"
    max_tokens = 500
    temperature = 0.00

    try:
        # Generate the response
        response = openai.ChatCompletion.create(
            model=model_engine,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature)

    except Exception as e:

        # Get the total number of requested tokens
        numbers = re.findall(r'\d+', e.args[0])
        total_tokens = numbers[1]

        # Create the log row
        log_row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_name,
            treatment,
            task,
            "",
            "",
            total_tokens,
            start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "",
            "",
            user_message["content"],
            "",
            e]

        # Write the log row to the log file
        with open(log_file_path, "a", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(log_row)

        # Print the error
        print(f"Error: {case_id} - {e}")
        continue

    # Inspect the response
    response_text = response.choices[0].message.content

    # Add a final new-line character
    # NOTE: This is necessary to fix a bug with the evaluation script
    response_text = response_text + "\n"

    # Write the verification to a file
    verifications_file_name = explanation_file_name
    verifications_file_path = verifications_folder_path + "/" + verifications_file_name
    with open(verifications_file_path, "w") as f:
        f.write(response_text)

        # Get the end time
    end_time = datetime.now()

    # Get the duration
    duration = end_time - start_time

    # Create the log row
    log_row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        model_name,
        treatment,
        task,
        response.usage["prompt_tokens"],
        response.usage["completion_tokens"],
        response.usage["total_tokens"],
        start_time.strftime("%Y-%m-%d %H:%M:%S"),
        end_time.strftime("%Y-%m-%d %H:%M:%S"),
        duration.total_seconds(),
        user_message["content"],
        response_text,
        ""]

    # Write the log row to the log file
    with open(log_file_path, "a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(log_row)

    # Display a status update
    print(f"Verification saved to {verifications_file_name}")

# Display the final status
print("All explanations verified.")