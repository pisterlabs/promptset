# RETURN_FILES FUNCTION
# Input: A well-described issue description
# Output: A list of file paths that are relevant

import requests
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI, ChatAnthropic
import time
import asyncio
from .treeutils import get_tree
import re
import os
from dotenv import load_dotenv
load_dotenv() 

llm = ChatOpenAI(temperature=0, model="gpt-4")
#llm2 = ChatAnthropic(model="claude-v1.3-100k", temperature=0)


def get_file_relevance(task, file_path, folderList):

    if any(folder in file_path for folder in folderList):
        return False

    if '.' in file_path:
        raw_url = f'https://raw.githubusercontent.com/{os.environ["GITHUB_ORG"]}/{os.environ["GITHUB_REPO"]}/{os.environ["GITHUB_BRANCH"]}/{file_path}'
        headers = {
            "Authorization": "Bearer " + os.environ['GITHUB_PAT']
        }
        response = requests.get(raw_url, headers=headers)
        if response.status_code == 200:
            # Extract the tree of paths from the JSON response
            file = response.text

            if len(file) > 500000:
                return False

            pattern = r'\.(c|cpp|java|py|rb|js|html|css|php|swift|go|cs|pl|lua|scala|kt|sh|sql|xml|md|txt|md|yaml|yml|ts|tsx)$'
            if not re.search(pattern, file_path, re.IGNORECASE):
                return False

            prompt_template = """
            You are an experienced developer chatbot working on a software issue. You can only respond "YES" or "NO". The task you're working on has been specified as "{task}". 

            Bearing in mind that you should keep good code style, is the file below relevant to solving this issue?

            Filename: {file_name}
            {raw_file}

            Please return *only* "YES" or "NO". You cannot return anything else. If the file is not editable code, return NO. If you're unsure, return YES.
            """

            if '/' in file_path:
                parts = file_path.split('/')
                file_name = parts[-1]
            else:
                file_name = file_path

            llm_chain = LLMChain.from_string(
                llm=llm, template=prompt_template)
            answer = llm_chain.predict(
                task=task, file_name=file_name, raw_file=file)
            if 'YES' in answer:
                return True
            else:
                return False
    else:
        # Directory
        prompt_template = """
            You are an experienced developer chatbot working on a software issue. You can only respond "YES" or "NO". The task you're working on has been specified as "{task}". 

            Are files in the folder below likely to be relevant to solving this issue?

            Project name: {project}
            Directory: {file_name}

            Please return *only* "YES" or "NO". You cannot return anything else. If you're unsure, return YES.
            """

        '''if '/' in file_path:
            parts = file_path.split('/')
            file_name = parts[-1]
        else:
            file_name = file_path'''

        llm_chain = LLMChain.from_string(llm=llm, template=prompt_template)
        answer = llm_chain.predict(
            task=task, file_name=file_path, project=os.environ["GITHUB_REPO"])
        if 'YES' in answer:
            return True
        else:
            folderList.append(file_path)
            return False


def return_files(task, verbose=False):
    files = get_tree()
    output = []
    ignored_folders = []
    for file in files:
        relevant = get_file_relevance(
            task=task, file_path=file, folderList=ignored_folders)
        if relevant:
            if '.' in file:
                output.append(file)

    print('Final output: ' + str(output))
    return output
