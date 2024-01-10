import os
import re
import langid
import openai
import requests
import json
from jira import JIRA

# Set up GitHub API
github_token = os.environ['GITHUB_TOKEN']
github_repo = os.environ['GITHUB_REPOSITORY']

# Set up Jira API
jira_base_url = os.environ['JIRA_BASE_URL']
jira_username = os.environ['JIRA_USERNAME']
jira_api_token = os.environ['JIRA_API_TOKEN']
jira_project_key = 'YOUR_JIRA_PROJECT_KEY'  # Replace with the actual Jira project key
jira_in_progress_status_id = 'YOUR_JIRA_IN_PROGRESS_STATUS_ID'  # Replace with the actual status ID for 'In Progress'
jira_resolved_status_id = 'YOUR_JIRA_RESOLVED_STATUS_ID'  # Replace with the actual status ID for 'Resolved'
jira = JIRA(server=jira_base_url, basic_auth=(jira_username, jira_api_token))

# Set up OpenAI API
openai.api_key = os.environ['OPENAI_API_KEY']

# Set up Judge0 API
judge0_base_url = 'https://api.judge0.com'
judge0_headers = {
    'Content-Type': 'application/json',
}

# Function to identify the programming language of the file using Judge0 API
def identify_language(file_content):
    endpoint = f"{judge0_base_url}/languages/"
    response = requests.get(endpoint, headers=judge0_headers)
    language_data = response.json()

    language_id = None
    for lang in language_data:
        if lang['name'].lower() in file_content.lower():
            language_id = lang['id']
            break

    return language_id

# Function to detect errors based on the programming language
def detect_errors(language_id, source_code):
    errors = []

    # Get the user-defined test cases from the file
    with open("user_test_cases.json", "r") as test_cases_file:
        test_cases = json.load(test_cases_file)

    # Run the code with each test case and check the output
    for i, test_case in enumerate(test_cases, start=1):
        input_data = test_case["input"]
        expected_output = test_case["output"]

        try:
            # Run the code with the input data using Judge0 API
            output = run_code_with_input(source_code, language_id, input_data)

            # Compare the output with the expected output
            if output.strip() != expected_output.strip():
                error_message = f"Test case {i} failed:\n"
                error_message += f"Input: {input_data}\n"
                error_message += f"Expected Output: {expected_output}\n"
                error_message += f"Actual Output: {output}\n"
                errors.append(error_message)
        except RuntimeError as e:
            # Handle compilation or runtime errors
            error_message = f"Test case {i} failed:\n"
            error_message += f"Input: {input_data}\n"
            error_message += f"Error: {str(e)}\n"
            errors.append(error_message)

    return errors

# Function to get code suggestions from OpenAI
def get_code_suggestions(error_context):
    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=error_context,
        max_tokens=100
    )
    return response.choices[0].text.strip()

# Function to create Jira ticket
def create_jira_ticket(system_details, errors):
    issue_dict = {
        'project': {'key': jira_project_key},
        'summary': 'Error in code file',  # Customize the summary as needed
        'description': f"System Details:\n{system_details}\n\nErrors:\n{errors}",
        'issuetype': {'name': 'Bug'}  # Use 'Bug' or the appropriate issue type
    }
    new_issue = jira.create_issue(fields=issue_dict)
    return new_issue

# Function to process files and detect errors
def process_file(file_path):
    with open(file_path, 'r') as file_content:
        content = file_content.read()

    # Identify the programming language using Judge0 API
    language_id = identify_language(content)

    # Detect errors based on the programming language
    errors = detect_errors(language_id, content)

    # Create Jira ticket with system details and errors
    system_details = f"File Path: {file_path}\nLanguage ID: {language_id}"
    new_ticket = create_jira_ticket(system_details, "\n".join(errors))
    print('Jira ticket created:', new_ticket.key)

    # Implement the loop for code suggestions from OpenAI and apply them to the code
    # For each error in the errors list, follow the steps mentioned earlier to get code suggestions,
    # apply the suggestions, and re-run the error detection process until all test cases pass.

# Main function
if __name__ == "__main__":
    # Replace 'YOUR_PROJECT_PATH' with the path to your project directory
    project_path = '/workspace'  # Assuming '/workspace' is the correct path in GitHub Codespaces

    # Iterate over files in the project directory
    for root, dirs, files in os.walk(project_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                process_file(file_path)
