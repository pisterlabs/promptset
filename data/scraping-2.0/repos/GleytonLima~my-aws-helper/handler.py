import json
import os

from dotenv import load_dotenv
from github import Auth
from github import Github
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def lambda_handler(event, context):
    user_input = event.get('user_input', 'default_instruction')

    openai_response = process_openai_input(user_input)

    instruction, parameters = process_openai_response(openai_response)

    if instruction == 'default_instruction':
        return {
            'statusCode': 200,
            'body': json.dumps(f'Instruction: {instruction}, Parameters: {parameters}')
        }

    if instruction == 'create_repo':
        return create_repo(parameters)

    return {
        'statusCode': 200,
        'body': json.dumps(f'Instruction: {instruction}, Parameters: {parameters}')
    }


def create_repo(parameters):
    handler_code = get_openai_code_example()

    repository_name = parameters.get('repo_name', 'default_repo_name')
    stack = parameters.get('stack', 'default_stack')

    github_token = os.environ['GITHUB_TOKEN']
    auth = Auth.Token(github_token)
    github = Github(auth=auth)
    user = github.get_user()

    repo = user.create_repo(repository_name, auto_init=False)

    readme_content = f"## Welcome to {repository_name}\nThis is a default README file for the {stack} stack."
    repo.create_file("README.md", "Initial commit", readme_content)

    repo.create_file("handler.py", "Initial commit", handler_code)

    return {
        'statusCode': 200,
        'body': f'Repositório "{repository_name}" criado com sucesso para a stack "{stack}".'
    }


def process_openai_input(user_input):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": 'You are an assistant that only responds with json. For example:  {"instruction": '
                           '"create_repo", "parameters": {"repo_name": "xpto", "stack": "aws_lambda"}}',
            },
            {
                "role": "user",
                "content": user_input,
            }
        ],
        model="gpt-3.5-turbo"
    )
    print(response.choices[0].message.content)

    if response.choices[0].message.content[0] != '{':
        return {'instruction': 'default_instruction', 'parameters': {}}

    return json.loads(response.choices[0].message.content)


def process_openai_response(openai_response):
    instruction = openai_response.get('instruction', 'default_instruction')
    parameters = openai_response.get('parameters', {})

    return instruction, parameters


def get_openai_code_example():
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": 'You are an assistant that only responds with Python code. No further explanations about '
                           'the content.',
            },
            {
                "role": "user",
                "content": "Generate a aws lambda python code example that only return a hello world",
            }
        ],
        model="gpt-3.5-turbo"
    )

    content = response.choices[0].message.content

    content = clear_code_markdown_code_blocks(content)

    print(content)

    with open('example.py', 'w') as f:
        f.write(content)

    return content


def clear_code_markdown_code_blocks(content):
    content = content.replace('```python', '').replace('```', '')
    content = content.split('\n')
    if content[0] == '':
        content.pop(0)
    if content[-1] == '':
        content.pop(-1)
    content = '\n'.join(content)
    return content


if __name__ == '__main__':
    event = {
        'user_input': 'Create a Github repository for the AWS Lambda with Python stack. The name of the repository '
                      'should be "sample-test-3".'
    }
    context = {}

    lambda_handler(event, context)
