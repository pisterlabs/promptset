from dataclasses import dataclass
import pprint
import json
import os
import openai

from gptpr.gitutil import BranchInfo
import gptpr.consolecolor as cc

TOKENIZER_RATIO = 4
MAX_TOKENS = 6000

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    print("Please set OPENAI_API_KEY environment variable.")
    exit(1)

openai.api_key = OPENAI_API_KEY

DEFAULT_PR_TEMPLATE = ('### Ref. [Link]\n\n## What was done?\n[Fill here]\n\n'
                       '## How was it done?\n[Fill here]\n\n'
                       '## How was it tested?\n[Fill here with test information from diff content or commits]')


def get_pr_template():
    pr_template = DEFAULT_PR_TEMPLATE

    try:
        github_dir = os.path.join(os.getcwd(), '.github')
        github_files = os.listdir(github_dir)
        pr_template_file = [f for f in github_files if f.lower().startswith('pull_request_template')][0]
        pr_template_file_path = os.path.join(github_dir, pr_template_file)

        with open(pr_template_file_path, 'r') as f:
            local_pr_template = f.read()

            if local_pr_template.strip() != '':
                print('Found PR template at:', pr_template_file_path)
                pr_template = local_pr_template
            else:
                print('Empty PR template at:', pr_template_file_path, 'using default template.')
    except Exception:
        print('PR template not found in .github dir. Using default template.')

    return pr_template


@dataclass
class PrData():
    branch_info: BranchInfo
    title: str
    body: str

    def to_display(self):
        return '\n'.join([
            f'{cc.bold("Repository")}: {cc.yellow(self.branch_info.owner)}/{cc.yellow(self.branch_info.repo)}',
            f'{cc.bold("Title")}: {cc.yellow(self.title)}',
            f'{cc.bold("Branch name")}: {cc.yellow(self.branch_info.branch)}',
            f'{cc.bold("Base branch")}: {cc.yellow("main")}',
            f'{cc.bold("PR Description")}:\n{self.body}',
        ])


functions = [
    {
        "name": "create_pr",
        "description": "Creates a Github Pull Request",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "PR title, following angular commit convention"
                },
                "description": {
                    "type": "string",
                    "description": "PR description"
                },
            },
            "required": ["title", "description"]
        },
    }
]


def get_pr_data(branch_info):
    system_content = '''
    You are a helpful assistant that helps a developer getting git diff changes, main commit,
    secondary commits and a Github PR template and returns the template filled with all required description
    and a PR title.
    In PR description, explain what was done, how it was done, tested, etc.
    If there are too many changes, you can list them in bullet points.
    '''

    messages = [
        {'role': 'system', 'content': system_content},
    ]

    if len(branch_info.main_commits) > 0:
        messages.append({'role': 'user', 'content': 'main commits: ' + '\n'.join(branch_info.main_commits)})
        messages.append({'role': 'user', 'content': 'secondary commits: ' + '\n'.join(branch_info.commits)})
    else:
        messages.append({'role': 'user', 'content': 'git commits: ' + '\n'.join(branch_info.commits)})

    messages.append({'role': 'user', 'content': 'PR template:\n' + get_pr_template()})

    current_total_length = sum([len(m['content']) for m in messages])

    if current_total_length / TOKENIZER_RATIO > MAX_TOKENS:
        raise Exception(f'Current total length {current_total_length} is greater than max tokens {MAX_TOKENS}')

    total_length_with_diff = current_total_length + len(branch_info.diff)
    if total_length_with_diff / TOKENIZER_RATIO > MAX_TOKENS:
        print('Total content length (with diff) is too big.', cc.red('Skipping diff content...'))
    else:
        messages.append({'role': 'user', 'content': 'Diff changes:\n' + branch_info.diff})

    response = openai.ChatCompletion.create(
        model='gpt-4-0613',
        messages=messages,
        functions=functions,
        function_call={'name': 'create_pr'},
        temperature=0,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    try:
        arguments = json.loads(response.choices[0].message.function_call.arguments)
    except Exception as e:
        print('Error to decode message:', e)
        print('Response message')
        pprint.pprint(response.choices[0].message)
        raise e

    return PrData(
        branch_info=branch_info,
        title=arguments['title'],
        body=arguments['description']
    )
