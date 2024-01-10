from github import Github
from dotenv import load_dotenv
import os
import datetime
import openai
import requests
from dateutil.parser import parse
from operator import itemgetter
import json
from markdown import markdown

class GithubCategorizer:
    def __init__(self, github_token, openai_key, system_message):
        self.github_token = github_token
        self.openai_key = openai_key
        self.system_message = system_message
        openai.api_key = openai_key

    def get_data(self, query_type, repo_owner, repo_name):
        print("Getting Github Data...")
        headers = {"Authorization": "Bearer " + self.github_token}
        reviews = ', reviews { totalCount }' if query_type == 'pullRequests' else ''

        query = f"""
        {{
            repository(name: "{repo_name}", owner: "{repo_owner}") {{
                {query_type}(last: 100, states: OPEN) {{
                    nodes {{
                        title
                        number
                        createdAt
                        updatedAt
                        url
                        comments {{
                            totalCount
                        }}
                        {reviews}
                        body
                    }}
                }}
            }}
        }}
        """
        request = requests.post('https://api.github.com/graphql', json={'query': query}, headers=headers)
        if request.status_code == 200:
            result = request.json()
            print("result: ", result)
            data = result['data']['repository'][query_type]['nodes']
            # Filter data updated in the last 7 days
            data = [item for item in data if (parse(item['updatedAt']).date() - datetime.date.today()).days < 7]
            # Sort data by combined comment and review count
            data.sort(key=lambda item: item['comments']['totalCount'] + item.get('reviews', {'totalCount': 0})['totalCount'], reverse=True)
            return data
        else:
            raise Exception("Query failed to run by returning code of {}. {}".format(request.status_code, query))

    def get_categorized_data(self, data):
        print("Categorizing Data...")
        data_index = {}
        # Create a dictionary of PR number to PR message
        data_messages = []
        for item in data:
            data_index[str(
                item['number'])] = f"#{item['number']} - [{item['title']}]({item['url']}) - Created at: {item['createdAt']} - Updated at: {item['updatedAt']} - Reviews: {item.get('reviews', {'totalCount': 0})['totalCount']} - Comments: {item['comments']['totalCount']}"
            simplified = f"#{item['number']} - {item['title']}"
            data_messages.append(simplified)

        # Join the messages with newline character
        data_messages = "\n".join(data_messages)
        user = {
            'role':
            'user',
            'content': data_messages
        }
        print("Calling OpenAI...")
        initial_response = ""
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=[self.system_message, user],
        )
        res = response.choices[0].message.content
        initial_response = res
        print("Got response from OpenAI: ", res)
        combined = {}
        for k, v in json.loads(res).items():
            temp = []
            for i in v:
                try:
                    temp.append(data_index[str(i)])
                except KeyError:
                    pass
            combined[k] = temp
        return combined

    def generate_notes(self, repo_url):
        print("Generating Notes...")
        repo_owner = repo_url.split("/")[-2]
        repo_name = repo_url.split("/")[-1]

        print("Getting PRs...")
        prs = self.get_data('pullRequests', repo_owner, repo_name)
        print("Categorizing PRs...")
        categorized_prs = self.get_categorized_data(prs)

        print("Getting Issues...")
        issues = self.get_data('issues', repo_owner, repo_name)
        print("Categorizing Issues...")
        categorized_issues = self.get_categorized_data(issues)

        md_string = "# Weekly Dev Call:\n\n"

        md_string += "## PRs:\n"
        for category, prs in categorized_prs.items():
            md_string += f"### {category}\n"
            for pr in prs:
                md_string += f"- {pr}\n"
            md_string += "\n"

        md_string += "## Issues:\n"
        for category, issues in categorized_issues.items():
            md_string += f"### {category}\n"
            for issue in issues:
                md_string += f"- {issue}\n"
            md_string += "\n"

        return md_string

if __name__ == "__main__":
    load_dotenv()
    github_token = os.environ['GITHUB_TOKEN']
    openai_key = os.environ['OPENAI_API_KEY']
    system_message = {
        'role': 'system',
        'content':
            f'''
            You are a Product Manager for a Bitcoin Open Source Project. 
            You are preparing a weekly dev call by organizing the open Pull Requests and Issues into useful categories.
            Output the categories and the PRs/Issues that belong to each category in the required output structure. Categories we normally use include (but not limited to): Consensus, On-Chain, Lightning Gateway, Devimint/Testing, and Misc, but make others too.
            Required Output Structure:
            {{
                "Category1": [
                    123,
                ],
                "Category2": [
                    456,
                    789
                ]
            }}
            '''
    }
    gc = GithubCategorizer(github_token, openai_key, system_message)
    repo_url = "https://github.com/fedimint/fedimint"
    md_string = gc.generate_notes(repo_url)
    with open("notes.md", "w") as f:
        f.write(md_string)
