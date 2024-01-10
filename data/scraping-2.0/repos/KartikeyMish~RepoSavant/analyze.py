import re
import os
import openai
from dotenv import load_dotenv
from github import Github

# Set up OpenAI API credentials
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_url(user_url):
    pattern = r"(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9_-]+)"
    match = re.search(pattern, user_url)
    if match:
        return match.group(0)
    else:
        return None

def get_user_repositories(username):
    client = Github(os.getenv('GITHUB_TOKEN'))
    user = client.get_user(username)
    print(user)
    repos = user.get_repos()
    repo_info = []
    for repo in repos:
        # If repo is a fork and no one starred it, skip it
        if repo.fork and repo.stargazers_count==0:
            continue
        print(repo.name)
        repo_info.append({
            "name": repo.name,
            "url": repo.html_url,
            "description": repo.description,
            "contributors": repo.get_contributors().totalCount,
            "size": repo.size,
            "language": repo.language,
            "topics": repo.get_topics(),
            "stars": repo.stargazers_count,
            "forks": repo.forks_count,
            "labels": repo.get_labels(),
            "issues": repo.get_issues(state="all"),
        })
    return repo_info


context = """
Utilizing the provided JSON data of a user's GitHub repositories, conduct a comprehensive analysis to pinpoint the repository 
with the highest level of technical complexity. Delve into a multitude of criteria, encompassing repository size, contributors
(striving for at least 4), tags, stargazer count as starts, pertinent topics, issues, pull requests, fork count, and the repository description. 
Once determined, furnish summary but concise (within 100 words) yet technically-informed summary elucidating the facets that contribute to its complexity.

( Most Important Must follow ) output format - all in one line & no double quotes around values ( IMOPORTANT ): name = repo_name , url = repo_url, justify  = analysis 
\
"""


def find_complex(json_data):
    # Generate analysis justification using GPT-3
    prompt = f"Analyzing repository {json_data}."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": context,
            },
        ],
        temperature=0,
        max_tokens=200,
    )
    analysis = response.choices[0].message.content
    return analysis


def nameRepoData(input_string):
    # Use regular expressions to extract name, url, and analysis
    # Define regular expressions to match the desired values
    name_pattern = r'name = ([^,]+)'
    url_pattern = r'url = ([^,]+)'
    justify_pattern = r'justify = (.+)'
    name_match = re.search(name_pattern, input_string)
    url_match = re.search(url_pattern, input_string)
    analysis_match = re.search(justify_pattern, input_string)
    if name_match and url_match and analysis_match:
        name = name_match.group(1)
        url = url_match.group(1)
        analysis = analysis_match.group(1)
        print("Complex repo : ",name)
        return name, url, cleanAnalysys(analysis)
    else:
        return "Error", "https://github.com", "Error"

def cleanAnalysys(input_text):
    # Find the position of the first opening parenthesis
    start_pos = input_text.find('(')
    end_pos = input_text.find(')')
    # If an opening parenthesis is found, extract the substring up to that position
    if start_pos != -1:
        output_text = input_text[:start_pos].strip()
        output_text += " "+input_text[end_pos+1:].strip()
    else:
        output_text = input_text
    return output_text
