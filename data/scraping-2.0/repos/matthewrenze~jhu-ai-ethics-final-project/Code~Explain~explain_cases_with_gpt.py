# Import libraries
import os
import re
import csv
from datetime import datetime
import openai

# Set the GPT model
model_name = "gpt-4"
treatment = "baseline"
task = "explain"

# Set file paths
root_folder_path = "C:/Users/Matthew/Dropbox/Personal/School/JHU/Ethics/Project"
system_prompt_file_path = root_folder_path + "/Data/Prompts/explain-cases-system-prompt.txt"
template_file_path = root_folder_path + "/Data/Templates/explain-cases-with-gpt-template.txt"
cases_folder_path = root_folder_path + "/Data/Cases"
example_cases_folder_path = root_folder_path + "/Data/Examples/Cases"
example_explanations_folder_path = root_folder_path + "/Data/Examples/Explanations/"
explainations_folder_path = root_folder_path + f"/Data/Explanations/{model_name}-{treatment}"
log_file_name = f"{model_name}-{treatment}-{task}.csv"
log_folder_path = root_folder_path + f"/Data/Logs"
log_file_path = log_folder_path + "/" + log_file_name

# Specify the number of examples to use (0-3)
number_of_examples = 3

# Set API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Display the initial status
print("Explaining all cases...")

# Create output directory if it doesn't exist
if not os.path.exists(explainations_folder_path):
    os.makedirs(explainations_folder_path)

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

# Write the header to the log file
with open(log_file_path, "a", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(log_header)

# Read the system prompt file
with open(system_prompt_file_path, "r") as system_prompt_file:
    system_prompt = system_prompt_file.read()

# Read the template file
with open(template_file_path, "r") as template_file:
    template = template_file.read()

# Create replacements values for placeholders in the system prompt
replacements = {"template": template}

# Replace the placeholders in the system prompt
system_prompt = system_prompt.format(**replacements)

# Read the example case files
example_cases = []
for file_name in os.listdir(example_cases_folder_path):
    file_path = example_cases_folder_path + "/" + file_name
    with open(file_path, "r") as file:
        example_cases.append(file.read())

# Read the example explanation files
example_explanations = []
for file_name in os.listdir(example_explanations_folder_path):
    file_path = example_explanations_folder_path + "/" + file_name
    with open(file_path, "r") as file:
        example_explanations.append(file.read())

# Read the case files
case_file_names = []
for file_name in os.listdir(cases_folder_path):
    case_file_names.append(file_name)

# *** HACK: ONLY PROCESS FAILED CASES ***
case_file_names = ["2758-jerome-traverso.txt", "7420-joshe-bittelman.txt"]

# For each case in the data
for case_file_name in case_file_names:

    # Display a status update
    print(f"Explaining case {case_file_name}...")

    # Get the start time
    start_time = datetime.now()

    # Get the case id
    case_id = case_file_name.split("-")[0]

    # Create the messages
    messages = []

    # Add the system prompt
    system_message = {"role": "system", "content": system_prompt}
    messages.append(system_message)

    # Add the example cases and explanations
    for i in range(number_of_examples):
        example_case = {"role": "user", "content": example_cases[i]}
        example_explanation = {"role": "assistant", "content": example_explanations[i]}
        messages.append(example_case)
        messages.append(example_explanation)

    # Read the case record
    case_file_path = cases_folder_path + "/" + case_file_name
    with open(case_file_path, "r") as case_file:
        case_record = case_file.read()

    # Add the user prompt
    user_message = {"role": "user", "content": case_record}
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

    # Get the response
    response_text = response.choices[0].message.content

    # Write the response to a text file
    explanation_file_name = case_file_name
    explanation_file_path = explainations_folder_path + "/" + explanation_file_name
    with open(explanation_file_path, "w") as f:
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
    print(f"Case {case_id} explained.")

# Display the final status
print("All cases explained.")