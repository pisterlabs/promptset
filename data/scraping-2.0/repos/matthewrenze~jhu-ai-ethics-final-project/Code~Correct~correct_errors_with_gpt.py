# Import libraries
import os
import re
import csv
from datetime import datetime
import openai

# Set the GPT model
model_name = "gpt-4"
treatment = "corrected-2"
task = "correct"

# Specify the number of examples to use (0-3)
# Note: Change this from 3 to 2 for GPT-3.5
number_of_examples = 3

# Set file paths
root_folder_path = "C:/Users/Matthew/Dropbox/Personal/School/JHU/Ethics/Project"
system_prompt_file_path = root_folder_path + "/Data/Prompts/correct-explanations-system-prompt.txt"
template_file_path = root_folder_path + "/Data/Templates/correct-explanations-template.txt"
cases_folder_path = root_folder_path + "/Data/Cases"
example_errors_folder_path = root_folder_path + "/Data/Examples/Errors"
example_explanations_folder_path = root_folder_path + "/Data/Examples/Explanations/"
explanations_folder_path = root_folder_path + f"/Data/Explanations/{model_name}-{treatment}"
errors_folder_path = root_folder_path + f"/Data/Errors/{model_name}-{treatment}"
corrections_folder_path = root_folder_path + f"/Data/Corrections/{model_name}-{treatment}"
log_file_name = f"{model_name}-{treatment}-{task}.csv"
log_folder_path = root_folder_path + f"/Data/Logs"
log_file_path = log_folder_path + "/" + log_file_name

# Set API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Specify the features to use
features = [
    "id",
    "name",
    "decile_score",
    "priors_count",
    "c_charge_degree",
    "juv_fel_count",
    "family_criminality",
    "criminal_attitude",
    "criminal_associates",
    "financial_problems",
    "substance_abuse",
    "noncompliance",
    "social_environment",
    "vocational"]

importance_features = [
    "priors_count_importance",
    "c_charge_degree_importance",
    "juv_fel_count_importance",
    "family_criminality_importance",
    "criminal_attitude_importance",
    "criminal_associates_importance",
    "financial_problems_importance",
    "substance_abuse_importance",
    "noncompliance_importance",
    "social_environment_importance",
    "vocational_importance"]

scaled_features = [
    "priors_count_importance_scaled",
    "c_charge_degree_importance_scaled",
    "juv_fel_count_importance_scaled",
    "family_criminality_importance_scaled",
    "criminal_attitude_importance_scaled",
    "criminal_associates_importance_scaled",
    "financial_problems_importance_scaled",
    "substance_abuse_importance_scaled",
    "noncompliance_importance_scaled",
    "social_environment_importance_scaled",
    "vocational_importance_scaled"]

# Display the initial status
print("Correcting all errors...")

# Create output directory if it doesn't exist
if not os.path.exists(corrections_folder_path):
    os.makedirs(corrections_folder_path)

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

# Read the example error files
example_errors = []
for file_name in os.listdir(example_errors_folder_path):
    file_path = example_errors_folder_path + "/" + file_name
    with open(file_path, "r") as file:
        example_errors.append(file.read())

# Read the example correction files
example_explanations = []
for file_name in os.listdir(example_explanations_folder_path):
    file_path = example_explanations_folder_path + "/" + file_name
    with open(file_path, "r") as file:
        example_explanations.append(file.read())

# Read the error files
errors_file_names = []
for file_name in os.listdir(errors_folder_path):
    errors_file_names.append(file_name)

# For each case in the data
for errors_file_name in errors_file_names:

    # Display a status update
    print(f"Correcting case {errors_file_name}...")

    # Get the start time
    start_time = datetime.now()

    # Get the case id
    case_id = errors_file_name.split("-")[0]

    # Create the messages
    messages = []

    # Add the system prompt
    system_message = {"role": "system", "content": system_prompt}
    messages.append(system_message)

    # Add the example cases and explanations
    for i in range(number_of_examples):
        example_case = {"role": "user", "content": example_errors[i]}
        example_explanation = {"role": "assistant", "content": example_explanations[i]}
        messages.append(example_case)
        messages.append(example_explanation)

    # Get the case file path
    case_file_name = errors_file_name
    case_file_path = cases_folder_path + "/" + case_file_name

    # Read the case file
    with open(case_file_path, "r") as case_file:
        case_record = case_file.read()

    # Get the explanation file path
    explanation_file_name = errors_file_name
    explanation_file_path = explanations_folder_path + "/" + explanation_file_name

    # Read the explanation file
    with open(explanation_file_path, "r") as explanation_file:
        explanation = explanation_file.read()

    # Get the errors file path
    errors_file_path = errors_folder_path + "/" + errors_file_name

    # Read the errors file
    with open(errors_file_path, "r") as errors_file:
        errors = errors_file.read()

    # Compose the user prompt
    user_prompt = case_record + "\n" \
        + "# Explanation\n" \
        + explanation + "\n\n" \
        + errors + "\n"

    # Add the user prompt
    user_message = {"role": "user", "content": user_prompt}
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

        # Display the error
        print(f"Error: {case_id} - {e}")
        continue

    # Get the response
    response_text = response.choices[0].message.content

    # Write the response to a text file
    correction_file_name = errors_file_name
    correction_file_path = corrections_folder_path + "/" + correction_file_name
    with open(correction_file_path, "w") as f:
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
    print(f"Case {case_id} corrected.")

# Display the final status
print("All cases corrected.")