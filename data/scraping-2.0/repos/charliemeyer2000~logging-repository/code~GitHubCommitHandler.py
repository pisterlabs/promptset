import requests
import datetime
import openai
import base64 
import json
import os

def lambda_handler(event, context):
    if 'body' in event:
        event = json.loads(event['body'])
    ghCommitHandler = GitHubCommitHandler(event)
    ghCommitHandler.handle_commit()
    return {
        'statusCode': 200,
        'body': json.dumps('Commit summary added to README.md file in logging-repository')
    }

class GitHubCommitHandler:
    def __init__(self, event):
        self.event = event
        self.payload = isinstance(event, str) and json.loads(event) or event
        self.commit_message = self.payload['head_commit']['message']
        self.commit_sha = self.payload['head_commit']['id']
        self.repository_committed_to = self.payload['repository']['name']
        self.files_modified = self.payload['head_commit']['modified']
        self.files_added = self.payload['head_commit']['added']
        self.files_removed = self.payload['head_commit']['removed']
        self.committer = self.payload['head_commit']['committer']['username']
        self.owner = "charliemeyer2000"
        self.GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
        openai.api_key = os.environ.get("OPENAI_API_KEY")

    def get_commit_data(self):
        url = f"https://api.github.com/repos/{self.owner}/{self.repository_committed_to}/commits/{self.commit_sha}"
        headers = {
            "Authorization": f"token {self.GITHUB_TOKEN}"
        }
        response = requests.get(url=url, headers=headers)
        data = response.json()
        return data
    
    # def get_num_tokens(self, string: str) -> int:
    #     """Returns the number of tokens in a text string for text-davinci-003"""
    #     encoding = tiktoken.encoding_for_model("text-davinci-003")
    #     num_tokens = len(encoding.encode(string))
    #     return num_tokens


    def generate_summary(self, code_added, code_removed, code_modified) -> str:
        """
        Takes in a string of code and returns a summary of the code

        :param commit_code: string of code
        :return: string of summary
        """

        # Generate summary using OpenAI
        prompt = f"""
        You are an AI tasked with generating a one-sentence summary of a git commit based on the files changed and the commit message.
        You are in a code pipeline, and any text you output will be taken
        directly and put into a markdown file downstream. You will be a reliable and trusted part of the pipeline. 
        You should not be using any special characters that are not just characters or numbers, as this will break
        the markdown file within this pipeline. You must perform this summary. Furthermore, you must add some humor
        to the summary, as this is a fun project, either by adding a joke or a funny comment.
        
        The files added are: {code_added}
        The files removed are: {code_removed}
        The files modified are: {code_modified}
        The commit message is: {self.commit_message}
        """

        # if the prompt is over 4096 tokens, 
        # truncate the prompt to 4096 tokens

        prompt_len = len(prompt)
        if (prompt_len > 4096 - 400):
            prompt = prompt[:4096 - 400]
            prompt += "..."

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.8,
            max_tokens=4096-prompt_len,
        )

        if(len(prompt) > 2000):
            return response.choices[0].text.strip() + "... Because more than 2000 characters were provided, the summary was truncated. Therefore, the summary may not entirely be accurate to all of the code & changes made in the commit."
        else:
            return response.choices[0].text.strip()

    def create_commit(self, commit_summary: str, commit_message: str, repository_committed_to: str, files_modified: any, committer: str) -> None:
        """
        Takes in a summary of the commit and adds it to the logs.md file in the logging-repository in my github account.

        :param commit_summary: summary of the commit
        :return: None
        """

        # Get the current logs.md file from the logging-repository
        url = "https://api.github.com/repos/charliemeyer2000/logging-repository/contents/logs.md"
        headers = {
            'Authorization': f'Bearer {self.GITHUB_TOKEN}',
            'X-GitHub-Api-Version': '2022-11-28'
        }

        # Get the current logs.md file
        response = requests.get(url=url, headers=headers)
        current_content = response.json()['content']
        current_content = base64.b64decode(current_content).decode('ascii', 'ignore')

        # Gets the current date and time
        current_date = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

        # Create a new addition to the logs.md file
        title = f"### {committer}: '{commit_message}' @ {current_date} to {repository_committed_to}\n"
        summary= ''
        
        # Add the GPT summary to the summary, make the summary a block quote of code block
        summary += f'\nGPT Summary: \n > {commit_summary}' + '\n' 
            

        # Add a h3 header to the TOP logs.md file with the commit message and the commit summary
        new_content = title + summary + '\n\n' + current_content

        # Encode the new content to base64 and avoiding emojis
        new_content = new_content.encode('ascii', 'ignore')
        new_content = base64.b64encode(new_content)

        # Update the logs.md file to be the old content + the new content
        data = {
            "message": f"Update logs.md for commit {commit_message}",
            "content": new_content.decode('ascii'),
            "sha": response.json()['sha']
        }

        print(f'commit summary: {commit_summary}')

        # Update the logs.md file
        # response = requests.put(url=url, headers=headers, data=json.dumps(data))

    def handle_commit(self):
        # data = self.get_commit_data()
        # lines_added, lines_removed = self.get_lines_of_code(data)
        commit_summary = self.generate_summary(code_added=self.files_added, code_removed=self.files_removed, code_modified=self.files_modified)
        self.create_commit(commit_summary=commit_summary, commit_message=self.commit_message, repository_committed_to=self.repository_committed_to, files_modified=self.files_modified, committer=self.committer)
        return 'Commit summary added to README.md file in logging-repository'