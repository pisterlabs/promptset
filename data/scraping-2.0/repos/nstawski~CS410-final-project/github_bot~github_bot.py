import ast
from argparse import ArgumentParser
from github import Github
import requests
import openai
import time

MODEL = "gpt-3.5-turbo"
def generate_prompt(pr_diff):
    """
    Prompt for OpenAI API, separated for readability

    :param pr_diff: a string representation of the GitHub PR diff
    :return: a prompt or OpenAI API
    """
    return f'''
        As an experienced software engineer, your task is to analyze the provided GitHub diff and provide structured comments on potential improvements in the pull request. 
        
        For each comment:

        1. Suggest a specific improvement, focusing on aspects such as wording, spelling corrections, coding style, or script enhancements. 
        2. Critical: please include a relevant code snippet that demonstrates your suggestion. Format this code snippet in backticks (`) for proper display as a GitHub code snippet.
        3. Ensure that the suggested code is valid Python code, matching the style and context of the original code in the diff. Make sure all comments are properly escaped.
        4. Crucial: Format your response as a Python list of tuples. Each tuple should be enclosed in round brackets and contain:
        - The file path as a string.
        - The line number as an integer where the suggestion applies, calculated for github using the diff hunk.
        - A detailed comment as a string, including a specific code snippet within backticks.
        - The corresponding diff hunk section as a string, accurately reflecting the location of the suggested change.
        The order of elements in the tuple is VERY important, please do not alter it.

        Your response must start with '[' and end with ']', containing only the list of tuples. Each tuple element should be properly formatted for Python, with all brackets correctly closed. The diff hunk should match the location of your suggestion precisely. Each comment's location should match the place that the comment is being applied to.
        The position is calculated using the diff. The comments should be made only on the changed portions of code (those that are denoted with "+" or "-" in the diff). Spelling errors etc. should not be checked for the lines with "-". Please make sure the comment position is calculated correctly using the diff hunks!

        Format your response like this example:
        [("file_path.py", 10, "Consider using list comprehension for efficiency: `[x for x in range(3)]`", "@@ -8,12 +8,15 @@"), ...]
        
        The rest after the semicolon will be the diff for your review:
        {pr_diff}
        '''

def get_last_pr_diff(settings, pr):
    """
    Retrieves the most recent diff from the GitHub pull request

    :param settings: a settings object generated from parsed arguments containing GitHub token, OpenAI API key and a repo name
    :param pr: a PyGitHub PR object
    :return: a string representation of the PR diff, or None if there was an error
    """
    # couldn't fetch the pr diff with the Github package for some reqson, using requests instead
    print(pr)
    api_url = f'https://api.github.com/repos/{settings.github_repo}/pulls/{pr.number}'
    headers = {'Authorization': f'token {settings.github_token}', 'Accept': 'application/vnd.github.diff'}
    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        print ('response.text', response.text)
        return response.text
    else:
        print(f'Error fetching diff: {response.status_code}, {response.text}')
        return None

    
def analyze_pr_diff(settings, pr_diff):
    """
    Analyzes a PR diff using the OpenAI API, then evaluates and parses the response into a list of tuples
    The function will retry on errors with OpenAI or during the output processing

    :param settings: a settings object generated from parsed arguments containing GitHub token, OpenAI API key and a repo name
    :param pr_diff: a string representation of the GitHub PR diff
    :return: a list of tuples with pull request comments, or an empty list if all attempts fail
    """

    openai.api_key = settings.openai_api_key
    max_attempts = 10
    retry_delay = 5  # seconds, delay between retries

    for attempt in range(max_attempts):
        print(f"Attempt {attempt + 1}/{max_attempts}")
        try:
            # Make the OpenAI API call
            print("Calling the OpenAI API")
            response = openai.ChatCompletion.create(
                model=MODEL,  
                messages=[
                    {"role": "system", "content": "You are an experienced software engineer well versed in Python"},
                    {"role": "user", "content": generate_prompt(pr_diff)}
                ]
            )
            print('Got response:', response)

            # Process the API response
            print("Processing the response...")
            feedback = response.choices[0].message.content.strip() if response.choices else ''
            formatted_feedback = f'[{feedback.strip()}]' if feedback.strip().startswith("(") else feedback.strip()
            cleaned_feedback = formatted_feedback.replace("\\\"", "\"").replace('\"', '"')
            print('Cleaned Feedback:', cleaned_feedback)

            # Evaluate the cleaned string as a Python object
            comments_to_add = ast.literal_eval(cleaned_feedback)
            return comments_to_add

        except (openai.error.APIError, ValueError, SyntaxError, Exception) as error:
            # Catching all the exceptions and errors that may happen within OpenAI call or subsequent parsing
            print(f'Error encountered: {error}')
            if attempt < max_attempts - 1:
                print("Retrying...")
                time.sleep(retry_delay)
            else:
                print("All retry attempts exhausted. Aborting.")
                return []

def add_comments_on_pull_request(pr, repo, pr_comments):
    """
    Adds given comments on a pull request

    :param pr: the pull request object
    :param repo: the GitHub repository object generated with PyGithub
    :param pr_comments: a list of tuples containing the comments to be added
    """

    # Fetch the commit object using its SHA
    commit = repo.get_commit(pr.head.sha)

    for file_path, position, comment, diff_hunk in pr_comments:

        print("file_path", file_path)
        print("position", position)
        print("comment", comment)
        print("commit", commit)
        print("diff_hunk", diff_hunk)

        pr.create_comment(
            body=comment, 
            commit_id=commit,  
            path=file_path, 
            position=position,  
        )
        # pr.create_review_comment(comment, commit, file_path, position)

def poll_prs_and_generate_comments(settings):
    """
    Polls open pull requests from a given repository on GitHub, generates comments using OpenAI and then adds them into a pull request

    :param settings: a settings object generated from parsed arguments containing GitHub token, OpenAI API key and a repo name
    """

    # set up
    github_client = Github(settings.github_token)
    repo = github_client.get_repo(settings.github_repo)

    print('Checking the repo for open pull requests...')
    open_pull_requests = repo.get_pulls(state='open', sort='created')

    for pr in open_pull_requests:
        print('Pull request number', pr.number)
        print('Pull request title', pr.title)
        print('Working on the diff...')

        pr_diff = get_last_pr_diff(settings, pr)

        pr_comments = analyze_pr_diff(settings, pr_diff)

        print(pr_comments)

        add_comments_on_pull_request(pr, repo, pr_comments)

if __name__ == "__main__":
    project_description='''
        This script is using the ChatGPT and GitHub API
        to add automated comments on a pull request,
        catch simple style errors and suggest improvements.
        Usage:
    '''
    arguments = {
        "github_token": "GitHub Personal Access Token",
        "openai_api_key": "OpenAI API Key",
        "github_repo": "GitHub username/repository name (use slash, no spaces)",
    }

    parser = ArgumentParser(project_description)
    
    for (key, value) in arguments.items():
        parser.add_argument(key, help=value)
        
    settings = parser.parse_args()
    poll_prs_and_generate_comments(settings)