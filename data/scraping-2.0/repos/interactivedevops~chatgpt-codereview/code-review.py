import os

import requests
import openai
from github import Github

# GitHub Keys
GH_TOKEN = os.getenv('GH_TOKEN') 
GITHUB_PR_ID = os.getenv('GITHUB_PR_ID') 
GITHUB_REPOSITORY = os.getenv('GITHUB_REPOSITORY') 

# Define your SonarQube server URL and authentication token
SONARQUBE_URL = os.getenv('SONARQUBE_URL') 
SONARQUBE_TOKEN = os.getenv('SONARQUBE_TOKEN') 
SONARQUBE_PROJECT_KEY = os.getenv('SONARQUBE_PROJECT_KEY') 

# Define your ChatGPT API endpoint and API key
CHATGPT_API_URL = os.getenv('CHATGPT_API_URL') 
CHATGPT_API_KEY = os.getenv('CHATGPT_API_KEY') 
CHATGPT_DEPLOYMENT_ID = os.getenv('CHATGPT_DEPLOYMENT_ID') 

# Function to fetch SonarQube issues
g = Github(login_or_token=GH_TOKEN)


def fetch_sonarqube_issues():
    issues_url = f'{SONARQUBE_URL}/api/issues/search'
    params = {
        'componentKeys': SONARQUBE_PROJECT_KEY,
        'severities': 'BLOCKER, CRITICAL, MAJOR, MINOR',
        'resolved': 'false',  # Get unresolved issues
        'ps': 100,  # Number of issues to retrieve (adjust as needed)
    }

    headers = {
        'Authorization': f'Bearer {SONARQUBE_TOKEN}',
    }

    try:
        response = requests.get(issues_url, params=params, headers=headers)
        response.raise_for_status()  # Raise an exception for bad responses (e.g., 404, 500)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching SonarQube issues: {e}")
        return None


# Function to generate code review comments using ChatGPT
def generate_code_review_comments(sonarqube_issues):
    if sonarqube_issues is None:
        return []  # Return an empty list if there was an issue with fetching SonarQube issues
    comments = []
    for issue in sonarqube_issues['issues']:
        # Extract relevant information from SonarQube issue
        issue_key = issue['key']
        issue_message = issue['message']

        # Prepare a message to send to ChatGPT
        chatgpt_input = f"SonarQube issue {issue_key}: {issue_message}"

        # Send the message to ChatGPT for review
        chatgpt_response = send_message_to_chatgpt(chatgpt_input)

        # Extract ChatGPT's response with suggested fixes
        suggested_fixes = chatgpt_response.get('choices', [{'message': 'No suggested fixes available.'}])[0]['message']

        file_name = issue['component']
        severity = issue['severity']
        line = issue['line']
        type = issue['type']
        add_pr_comment(file_name, severity, line, type, suggested_fixes)

    return comments


# add comment in PR
def add_pr_comment(file_name, severity, line, type, message):
    # repo = g.get_repo(os.getenv('GITHUB_REPOSITORY'))
    repo = g.get_repo(GITHUB_REPOSITORY)
    pull_request = repo.get_pull(int(GITHUB_PR_ID))

    if check_fileexist_in_commit(pull_request, file_name):
        pr_comment = f"<b> File name: </b> {file_name} \n<b> Severity: </b> {severity} \n<b> Line no: </b> {line} \n<b> " \
                     f"Type: </b> {type} \n\n <b>Review comment by ChatGPT about `{file_name}`</b>:\n {message['content']}"
        pull_request.create_issue_comment(pr_comment)


def check_fileexist_in_commit(pull_request, file_name):
    commits = pull_request.get_commits()
    actual_filename = file_name.split(':')[1]
    for commit in commits:
        files = commit.files
        for file in files:
            filename = file.filename
            if filename.__contains__(actual_filename):
                return True
    return False


# Function to send a message to ChatGPT for review
def send_message_to_chatgpt(message):
    headers = {
        'Authorization': f'Bearer {CHATGPT_API_KEY}',
        'Content-Type': 'application/json',
    }
    analysis_input = f"Please review the following application code technically: {message}"
    data = [{"role": "system",
             "content": "You are a Senior Developer/Engineer, provide the fix for below SonarQube findings for "
                        "developers to fixed it in their code"},
            {"role": "system", "content": analysis_input}]

    try:
        openai.api_key = CHATGPT_API_KEY
        openai.api_base = CHATGPT_API_URL
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"

        analysis_response = openai.ChatCompletion.create(
            engine=CHATGPT_DEPLOYMENT_ID,
            deployment_id=CHATGPT_DEPLOYMENT_ID,
            messages=data,
            temperature=0
        )
        return analysis_response.to_dict()
    except requests.exceptions.RequestException as e:
        print(f"Error sending message to ChatGPT: {e}")
        return {'choices': [{'text': 'Error communicating with ChatGPT.'}]}


if __name__ == "__main__":
    sonarqube_issues = fetch_sonarqube_issues()
    generate_code_review_comments(sonarqube_issues)
