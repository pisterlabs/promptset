import os
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv() 
import requests


llm = ChatOpenAI(temperature=0, model="gpt-4")


def generate_todos(task, files):
    bytes_dict = {}
    for file in files:
        raw_url = f'https://raw.githubusercontent.com/{os.environ["GITHUB_ORG"]}/{os.environ["GITHUB_REPO"]}/{os.environ["GITHUB_BRANCH"]}/{file}'
        headers = {
            "Authorization": "Bearer " + os.environ['GITHUB_PAT']
        }
        response = requests.get(raw_url, headers=headers)
        print(response.text)

        if response.status_code == 200:
            prompt_template = """
            You are an assistant developer chatbot. You are working on the below task:

            {task}

            You have a file to work on, at the path {path}. If changing the file or functions within it would help execute the task, leave TODO comments within the code suggesting changes to another developer. DO NOT edit ANY actual code or make ANY changes yourself. The code should work identically to before. Do not return anything other than the updated code. Here is the file:

            {file}
            """

            llm_chain = LLMChain.from_string(llm=llm, template=prompt_template)
            answer = llm_chain.predict(
                task=task, path=file, file=response.text)

            bytes_dict[str(file)] = bytes(answer, 'utf-8')

    prompt_template = """You are a developer coming up with names for branches.

    Examples of branches:
    - name=sagekhanuja and task=fixing-bugs, branch name=sk/fixing-bugs
    - name=sprucecampbell and task=removing button from the bottom of the page, branch name=sc/removing-button.


    """
    return bytes_dict
