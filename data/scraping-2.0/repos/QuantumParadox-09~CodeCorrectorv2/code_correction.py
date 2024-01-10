import os
import re
import json
import langid
import openai
import requests
from jira import JIRA
from github import Github

# Set up GitHub API
github_token = os.environ['GITHUB_TOKEN']
github_repo = os.environ['GITHUB_REPOSITORY']
g = Github(github_token)
repo = g.get_repo(github_repo)

# Set up Jira API
jira_base_url = os.environ['JIRA_BASE_URL']
jira_username = os.environ['JIRA_USERNAME']
jira_api_token = os.environ['JIRA_API_TOKEN']
jira = JIRA(server=jira_base_url, basic_auth=(jira_username, jira_api_token))

# Set up OpenAI API
openai.api_key = os.environ['OPENAI_API_KEY']

# Set up Judge0 API
judge0_base_url = 'https://api.judge0.com'
judge0_api_token = os.environ['JUDGE0_API_TOKEN']
judge0_headers = {
    'Authorization': f'Token {judge0_api_token}',
    'Content-Type': 'application/json'
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
def detect_errors(language_id, file_content):
    errors = []

    # Implement dynamic error detection logic using external code analysis API for the given language
    # For example, use a linter, static analyzer, or compiler API to detect errors in the code

    return errors

# Function to get code suggestions from OpenAI
def get_code_suggestions(error_context):
    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=error_context,
        max_tokens=100
    )
    return response.choices[0].text.strip()

# Function to create Jira tickets
def create_jira_ticket(error_details):
    issue_dict = {
        'project': {'key': 'YOUR_JIRA_PROJECT_KEY'},
        'summary': 'Error in code file',  # Customize the summary as needed
        'description': error_details,
        'issuetype': {'name': 'Bug'}  # Use 'Bug' or the appropriate issue type
    }
    new_issue = jira.create_issue(fields=issue_dict)
    return new_issue

# Function to run code with input data using Judge0 API
def run_code_with_input(input_data, language_id):
    endpoint = f"{judge0_base_url}/submissions/?base64_encoded=false&wait=true&fields=stdout"
    payload = {
        "source_code": updated_content,
        "language_id": language_id
    }
    response = requests.post(endpoint, headers=judge0_headers, json=payload)
    run_output = response.json().get('stdout', '')
    return run_output

# Function to process files and detect errors
def process_file(file_path):
    with open(file_path, 'r') as file_content:
        content = file_content.read()

    # Identify the programming language using Judge0 API
    language_id = identify_language(content)

    # Detect errors based on the programming language using external code analysis API
    errors = detect_errors(language_id, content)

    # Read user-defined test cases from the separate file "user_test_cases.json"
    with open('user_test_cases.json', 'r') as test_cases_file:
        test_cases_data = json.load(test_cases_file)

    # Iterate over user-defined test cases and run the code with input data using Judge0 API
    for test_case in test_cases_data:
        input_data = test_case['input']
        expected_output = test_case['output']

        # Run code with input data using Judge0 API
        run_output = run_code_with_input(input_data, language_id)

        # Compare the output with the expected output from the user-defined test cases
        if run_output.strip() == expected_output.strip():
            test_case['passed'] = True
        else:
            test_case['passed'] = False

    # Iterate over detected errors and get code suggestions from OpenAI
    for error in errors:
        # Create Jira ticket with error details and code suggestion
        error_details = f"Error in file: {file_path}\n{error}"
        new_ticket = create_jira_ticket(error_details)
        print('Jira ticket created:', new_ticket.key)

        # Test the code against user-defined test cases
        test_cases_passed = all(test_case['passed'] for test_case in test_cases_data)
        while not test_cases_passed:
            # Get code suggestions from OpenAI
            error_context = f"{file_path} - {error}\n{content}"
            code_suggestion = get_code_suggestions(error_context)

            # Apply the code suggestion to the file content
            updated_content = content.replace(error, code_suggestion)

            # Run code with input data using Judge0 API
            for test_case in test_cases_data:
                input_data = test_case['input']
                expected_output = test_case['output']

                # Run code with input data using Judge0 API
                run_output = run_code_with_input(input_data, language_id)

                # Compare the output with the expected output from the user-defined test cases
                if run_output.strip() == expected_output.strip():
                    test_case['passed'] = True
                else:
                    test_case['passed'] = False

            test_cases_passed = all(test_case['passed'] for test_case in test_cases_data)

            content = updated_content

    # Update the file with the corrected content
    with open(file_path, 'w') as file_content:
        file_content.write(content)

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
