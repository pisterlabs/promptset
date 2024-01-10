from langchain import LLMChain
from langchain.chat_models import ChatOpenAI, ChatAnthropic
import os
from .treeutils import get_tree, print_tree
import requests
import re
from dotenv import load_dotenv
load_dotenv() 

llm = ChatOpenAI(temperature=0, model="gpt-4")
llm2 = ChatAnthropic(model="claude-v1.3-100k", temperature=0)
llm3 = ChatOpenAI(temperature=0)

raw_url = f'https://raw.githubusercontent.com/{os.environ["GITHUB_ORG"]}/{os.environ["GITHUB_REPO"]}/{os.environ["GITHUB_BRANCH"]}/README.md'
response = requests.get(raw_url, headers=None)

if response.status_code == 200:
    project_description = response.text
else:
    project_description = "An open source software project."

def detail_task(task, files):
    fileString = ''
    for file in files:
        raw_url = f'https://raw.githubusercontent.com/{os.environ["GITHUB_ORG"]}/{os.environ["GITHUB_REPO"]}/{os.environ["GITHUB_BRANCH"]}/{file}'
        response = requests.get(raw_url, headers=None)

        if response.status_code == 200:
            #files += '\n\nFilename: ' + file + '\n' + response.text
    
            prompt_template = """
            You are an experienced developer chatbot. You are working on the below task:

            {task}

            You have these files to work on:

            {files}

            Propose a detailed plan, as if explaining to a pair programmer, of how you would solve this issue. If necessary, propose function and class names and changes for each file. Only return your proposed solution. 
            """

            llm_chain = LLMChain.from_string(llm=llm2, template=prompt_template)
            answer = llm_chain.predict(task=task, files=fileString)
            return answer

def explain_task(task):
    prompt_template = """
        You are an experienced developer chatbot working on a software issue. The issue you're working on is 
        
        {task}
        
        You can use the below context:

        Project description: 
        {description}
        File tree:
        {tree}

        Outline how you would solve this issue at a high level, as you would if creating a Github Issue to give to another chatbot developer on the project. Only return your proposed solution. Please be extremely concise. Do not mention any specific code or URLS.
        """

    llm_chain = LLMChain.from_string(llm=llm, template=prompt_template)
    answer = llm_chain.predict(
        task=task, description=project_description, tree=print_tree(get_tree()))
    
    prompt_template2 = """
        How many hours do you think this coding task would take? It's fine to guess, just return a specific number. Return ONLY a number, with no additional text, not even punctuation.

        {task}
        """

    llm_chain = LLMChain.from_string(llm=llm3, template=prompt_template2)
    answer2 = llm_chain.predict(
        task=answer)
    
    match = re.search(r'\d+', answer2)
    hours = 10
    if match:
        hours = int(match.group())

    codeGen = False
    if hours < 10:
        codeGen = True
    return {'text': answer, 'codeGen': codeGen}
