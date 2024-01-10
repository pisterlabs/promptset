#!/usr/bin/env python3

from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
import openai
import pandas as pd
import re

def create_prompt(code):
    prompt = r'''
    Please answer the following questions regarding the code. Please indicate the question each portions of the response is answering. Please do not use any additional sentences providing explanation other that what is asked by the question. Each question should be answerable in less than 20 words. Do not answer in complete sentences. Do not repeat the question back to me. There is an example below for what a response should look like. 
    1) Yes or No: Does the ChatCompletions API call use both a system and user for the conversation? 
    2) Is the user message static or dynamic? A static message refers to a constant string, while a dynamic message uses user input and variables to create the string. Dynamic inputs are variables taken from the command line that are inserted into the message string, typically by using format strings or concatenation. Function parameters are not dynamic tokens.
    3) If the prompt is in English, how many static words are in the user content prompt? If it is not in English, return "Not English".
    4) Where are the dynamic tokens located within the prompt template? Are they located at the beginning of the template, in the middle, or at the end? Return N/A only if the user prompt is static.
    5) How many steps are there between the command line input and the final dynamic tokens? What types of steps are they (i.e., concatenation, slicing, random generator)? If there are no steps, return 0. Return N/A only if the user prompt is static.
    //which part is dynamic vs static 
    Example 1: 
    #!/usr/bin/env python3

    import openai
    import sys
    import os
    import configparser

    # Get config dir from environment or default to ~/.config
    CONFIG_DIR = os.getenv('XDG_CONFIG_HOME', os.path.expanduser('~/.config'))
    API_KEYS_LOCATION = os.path.join(CONFIG_DIR, 'openaiapirc')

    # Read the organization_id and secret_key from the ini file ~/.config/openaiapirc
    # The format is:
    # [openai]
    # organization_id=<your organization ID>
    # secret_key=<your secret key>

    # If you don't see your organization ID in the file you can get it from the
    # OpenAI web site: https://openai.com/organizations
    def create_template_ini_file():
        """
        If the ini file does not exist create it and add the organization_id and
        secret_key
        """
        if not os.path.isfile(API_KEYS_LOCATION):
            with open(API_KEYS_LOCATION, 'w') as f:
                f.write('[openai]\n')
                f.write('organization_id=\n')
                f.write('secret_key=\n')
                f.write('model=gpt-3.5-turbo-0613\n')

            print('OpenAI API config file created at {}'.format(API_KEYS_LOCATION))
            print('Please edit it and add your organization ID and secret key')
            print('If you do not yet have an organization ID and secret key, you\n'
                'need to register for OpenAI Codex: \n'
                    'https://openai.com/blog/openai-codex/')
            sys.exit(1)


    def initialize_openai_api():
        """
        Initialize the OpenAI API
        """
        # Check if file at API_KEYS_LOCATION exists
        create_template_ini_file()
        config = configparser.ConfigParser()
        config.read(API_KEYS_LOCATION)

        openai.organization_id = config['openai']['organization_id'].strip('"').strip("'")
        openai.api_key = config['openai']['secret_key'].strip('"').strip("'")

        if 'model' in config['openai']:
            model = config['openai']['model'].strip('"').strip("'")
        else:
            model = 'gpt-3.5-turbo'

        return model

    model = initialize_openai_api()

    cursor_position_char = int(sys.argv[1])

    # Read the input prompt from stdin.
    buffer = sys.stdin.read()
    prompt_prefix = 'Here is the code: #!/bin/zsh\n\n' + buffer[:cursor_position_char]
    prompt_suffix = buffer[cursor_position_char:]
    full_command = prompt_prefix + prompt_suffix
    response = openai.ChatCompletion.create(model=model, messages=[
        {
            "role":'system',
            "content": "You are a zsh shell expert, please help me complete the following command, you should only output the completed command, no need to include any other explanation",
        },
        {
            "role":'user',
            "content": full_command,
        }
    ])
    completed_command = response['choices'][0]['message']['content']

    sys.stdout.write(f"\n{completed_command.replace(prompt_prefix, '', 1)}")
    An example response to Example 1 would be as follows:
    1) Yes.
    2) Dynamic. The dynamic variable is `buffer`.
    3) 5.
    4) End.
    5) There are 3 total operations: 2 slicing operations of the command line input, 1 concatenation operation on the two results. 

    Example 2: 
    import sys, re
    from pathlib import Path
    from os import path

    sys.path.append(str(Path(__file__).parent.parent.parent))

    import g4f

    def read_code(text):
        if match := re.search(r"```(python|py|)\n(?P<code>[\S\s]+?)\n```", text):
            return match.group("code")
        
    path = input("Path: ")

    with open(path, "r") as file:
        code = file.read()

    prompt = f"""
    Improve the code in this file:
    ```py
    {code}
    ```
    Don't remove anything.
    Add typehints if possible.
    Don't add any typehints to kwargs.
    Don't remove license comments.
    """

    print("Create code...")
    response = []
    for chunk in g4f.ChatCompletion.create(
        model=g4f.models.gpt_35_long,
        messages=[{"role": "user", "content": prompt}],
        timeout=300,
        stream=True
    ):
        response.append(chunk)
        print(chunk, end="", flush=True)
    print()
    response = "".join(response)

    if code := read_code(response):
        with open(path, "w") as file:
            file.write(code)
    An example response to Example 2 would be as follows:
    1) No.
    2) Dynamic. The dynamic variables for the user prompt is `path`.
    3) 23.
    4) Middle.
    5) There are 2 steps between the command line input and the final dynamic tokens. The steps are: reading the code from the file and inserting the code into the prompt.

    Example 3: 
    import openai

    # openai.log = "debug"
    openai.api_key = "sk-"
    openai.api_base = "https://api.chatanywhere.com.cn/v1"



    # 非流式响应
    # completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello world!"}])
    # print(completion.choices[0].message.content)

    def gpt_35_api_stream(messages: list):
        """为提供的对话消息创建新的回答 (流式传输)

        Args:
            messages (list): 完整的对话消息
            api_key (str): OpenAI API 密钥

        Returns:
            tuple: (results, error_desc)
        """
        try:
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=messages,
                stream=True,
            )
            completion = {'role': '', 'content': ''}
            for event in response:
                if event['choices'][0]['finish_reason'] == 'stop':
                    print(f'收到的完成数据: {completion}')
                    break
                for delta_k, delta_v in event['choices'][0]['delta'].items():
                    print(f'流响应数据: {delta_k} = {delta_v}')
                    completion[delta_k] += delta_v
            messages.append(completion)  # 直接在传入参数 messages 中追加消息
            return (True, '')
        except Exception as err:
            return (False, f'OpenAI API 异常: {err}')

    if __name__ == '__main__':
        messages = [{'role': 'user','content': '鲁迅和周树人的关系'},]
        print(gpt_35_api_stream(messages))
        print(messages)

    An example response to Example 3 would be as follows:
    1) No.
    2) Static. 
    3) Not English.
    4) N/A.
    5) N/A.
    '''
    return f'The following is a piece of code: {code}\n' + prompt

def transform_answers(answers):
    res = (0, 0, [], 0, 0, 0)
    try: 
        q1, q2, q3, q4, q5 = answers
    except:
        raise Exception(f'unable to transform answers: {answers}')
    
    if q1 == 'Yes.':
        res[0] = 1
    elif q1 == 'No.':
        pass
    else:
        raise Exception(f'invalid answers to question 1: {q1}')
    
    q2_split = q2.split(' ')
    if q2_split[0] == 'Dynamic.':
        res[1] = 1
    elif q2_split[0] == 'Static.':
        pass
    else:
        raise Exception(f'invalid answers to question 2: {q2}')
    for word in q2_split:
        if '`' in word:
            res[2].append(word)

    if q3 == 'Not English.':
        res[3] = -1
    else:
        res[3] = int(q3[:-1])

    if q4 == 'Beginning.':
        res[4] = 0
    elif q4 == 'Middle.':
        res[4] = 1
    elif q4 == 'End.':
        res[4] = 2
    else:
        raise Exception(f'invalid answers to question 4: {q4}')
    
    steps_pattern = r'(\d+)\s+steps'
    steps_match = re.search(steps_pattern, q5)
    res[5] = int(steps_match.group(1))
    
    return res

with open('openai_token', 'r') as f:
    OPENAI_KEY = f.readline().strip()
openai.api_key = OPENAI_KEY

with open('repos/repos.txt', 'r') as f:
    lines = f.readlines()

df = pd.DataFrame(columns=['repo_name', 'repo_path', 'system and user?', 'dynamic?', 'dynamic_vars', 'words', 'beg(0), middle(1), end(2)', 'steps'])

for line in lines:
    for i in range(5):
        fn, repo_name, repo_path = line.strip()[1:-1].split(', ')
        fn = fn[1:-1]
        with open(f'repos/{fn}', 'r') as f:
            code = f.read()
            completion = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                temperature=0.2,
                messages=[
                    {'role': 'system', 'content': 'You are a helpful code tracer.'}, 
                    {'role': 'user', 'content': create_prompt(code)}
                ]
            )
            print(completion.choices[0].message.content)
    exit(0)

# for i, line in enumerate(lines):
#     fn, repo_name, repo_path = line.strip()[1:-1].split(', ')
#     fn = fn[1:-1]
#     if fn[-6:] == '.ipynb':
#         # skip jupyter notebooks for now because they are too large
#         continue
#     repo_name = repo_name[1:-1]
#     repo_path = repo_path[1:-1]
#     print(f'====={repo_name}/{repo_path}======')

#     with open(f'repos/{fn}', 'r') as f:
#         code = f.read()
#     completion = openai.ChatCompletion.create(
#         model='gpt-3.5-turbo',
#         temperature=0.2,
#         messages=[
#             {'role': 'system', 'content': 'You are a helpful code tracer.'}, 
#             {'role': 'user', 'content': create_prompt(code)}
#         ]
#     )
#     print(completion.choices[0].message.content)
    
#     text = completion.choices[0].message.content

#     pattern = r'\d+\)\s(.*?)(?=(\d+\)|$))'
#     matches = re.findall(pattern, text)
#     answers = [match[0].strip() for match in matches if match[0].strip()]

#     try:
#         print(transform_answers(answers))
#     except: 
#         print('exiting here')
#         exit(0)
#         continue
#     exit(0)

# run multiple times
# print the template
# holdout set with different set