import os
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI, ChatAnthropic
import asyncio
import requests


llm = ChatOpenAI(temperature=0, model="gpt-4")

async def generate_code(task, files):
    bytes_dict = {}
    tasks = [fill_file(file, task, bytes_dict) for file in files]
    await asyncio.gather(*tasks)
    return bytes_dict


async def fill_file(file, task, bytes_dict):
    raw_url = f'https://raw.githubusercontent.com/{os.environ["GITHUB_ORG"]}/{os.environ["GITHUB_REPO"]}/{os.environ["GITHUB_BRANCH"]}/{file}'
    headers = {
        "Authorization": "Bearer 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"
    }
    response = requests.get(raw_url, headers=headers)
    print(response.text)

    if response.status_code == 200:
        prompt_template = """
        You are an experienced developer chatbot. You are working on the below task:

        {task}

        You have a file to work on, at the path {path}. If changing the file or functions within it would help execute the task, change the code. Try to stick to the plan as much as possible, and comment your code thoroughly with reasoning. Do not return anything other than the updated code. If no changes are needed, return the original file. Here is the file:

        {file}
        """

        llm_chain = LLMChain.from_string(llm=llm, template=prompt_template)
        answer = llm_chain.predict(
            task=task, path=file, file=response.text)

        bytes_dict[str(file)] = bytes(answer, 'utf-8')