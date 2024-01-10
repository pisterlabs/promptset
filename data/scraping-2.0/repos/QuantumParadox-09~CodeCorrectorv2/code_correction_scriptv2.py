import os
import re
import langid
import openai
import requests
from jira import JIRA
from github import Github
import json

# Function to read the user-defined test cases from the "user_test_cases.json" file
def read_user_test_cases():
    with open("user_test_cases.json", "r") as file:
        test_cases = json.load(file)
    return test_cases

# Function to identify the programming language of the file using Judge0 API
def identify_language(file_content):
    # Implement logic to identify the programming language using Judge0 API
    endpoint = f"{judge0_base_url}/languages/"
    response = requests.get(endpoint, headers=judge0_headers)
    language_data = response.json()

    language_id = None
    for lang in language_data:
        if lang['name'].lower() in file_content.lower():
            language_id = lang['id']
            break

    return language_id 

# Function to detect errors based on the programming language using Judge0 API
def detect_errors(language, file_content):
    errors = []

    # Set up Judge0 API
    judge0_base_url = 'https://api.judge0.com'
    judge0_headers = {
        'Content-Type': 'application/json',
    }

    # Compile the code using Judge0 API
    compile_url = f"{judge0_base_url}/submissions?wait=true"
    payload = {
        "source_code": file_content,
        "language_id": language
    }
    response = requests.post(compile_url, headers=judge0_headers, json=payload)
    submission_details = response.json()

    # Check if there are any compile errors
    if 'compile_output' in submission_details:
        errors.append(submission_details['compile_output'])
    else:
        # Run the code using Judge0 API
        run_url = f"{judge0_base_url}/submissions/{submission_details['id']}/execute?wait=true"
        response = requests.get(run_url, headers=judge0_headers)
        submission_details = response.json()

        # Check if there are any runtime errors
        if 'stderr' in submission_details:
            errors.append(submission_details['stderr'])

    return errors

# Function to check if the output of the code matches the expected output for each test case
def check_test_cases(output, test_cases):
    # Implement the logic to compare the output with the expected output for each test case
    for test_case in test_cases:
        input_data = test_case['input']
        expected_output = test_case['output']

        # Run the code with the input data to get the output
        run_output = run_code_with_input(input_data, test_cases)

        # Compare the output with the expected output for the current test case
        if run_output.strip() != expected_output.strip():
            # Test case failed
            return False

    # All test cases passed
    return True


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

# Main function
def main():
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
    judge0_headers = {
        'Content-Type': 'application/json',
    }

    # Replace 'YOUR_PROJECT_PATH' with the path to your project directory
    project_path = '/workspace'  # Assuming '/workspace' is the correct path in GitHub Codespaces

    # Iterate over files in the project directory
    for root, dirs, files in os.walk(project_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)

                # Read the file content
                with open(file_path, 'r') as file_content:
                    content = file_content.read()

                # Identify the programming language using Judge0 API
                language_id = identify_language(content)

                # Detect errors based on the programming language using Judge0 API
                errors = detect_errors(language_id, content)

                # Iterate over detected errors and get code suggestions from OpenAI
                for error in errors:
                    # Create Jira ticket with error details and code suggestion
                    error_details = f"Error in file: {file_path}\n{error}"
                    new_ticket = create_jira_ticket(error_details)
                    print('Jira ticket created:', new_ticket.key)

                    # Test the code against user-defined test cases
                    test_cases = read_user_test_cases()
                    test_cases_passed = False
                    while not test_cases_passed:
                        # Get code suggestions from OpenAI
                        error_context = f"{file_path} - {error}\n{content}"
                        code_suggestion = get_code_suggestions(error_context)

                        # Apply the code suggestion to the file content
                        updated_content = content.replace(error, code_suggestion)

                        # Run the updated code on Judge0 with user-defined test cases
                        run_output = run_code_with_input(updated_content, test_cases)

                        # Check if the output matches the expected output for each test case
                        test_cases_passed = check_test_cases(run_output, test_cases)

                        # If the test cases fail, fetch new code suggestions
                        if not test_cases_passed:
                            content = updated_content

                        # If the test cases pass, break the loop and update the file
                        else:
                            # Update the file with the corrected content
                            with open(file_path, 'w') as file_content:
                                file_content.write(updated_content)

if __name__ == "__main__":
    main()
