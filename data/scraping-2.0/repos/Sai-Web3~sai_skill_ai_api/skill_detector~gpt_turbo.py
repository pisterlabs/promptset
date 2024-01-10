import os
import openai
# ../sai_skill_api/._important_keys.pyの内容を読み込む
from sai_skill_api._important_keys import *


# Load environment variables from .env file

def get_chatgpt35_turbo_response(input_text: str) -> str:
    summary = openai.ChatCompletion.create(
        engine=OPENAI_CHAT_ENGINE, # engine = "deployment_name".
        messages=[
            {"role":"system","content":"You are the lead engineer who performs the technology selection for the client's requirements.. Please itemize the names of 30 specific skills. \ne.g.) C, C#, Javascript. \nYou MUST return English.\nSkills should preferably be a reference to a programming language or framework."},{"role":"user","content":"We want to create a platform for music that We automatically change the data of music submitted by users to the best data for streaming and distribute it."},{"role":"assistant","content":"Skill needed: JavaScript, Python, C, React, Vue, Next.js, Nest.js, MySQL, HTML, CSS, R, Django, Rust, Ruby, Ruby on Rails"},
            {"role": "user", "content":  input_text},
        ],
        temperature=0.1,
        max_tokens=2045,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    print("summary")
    print(summary)
    summary = summary['choices'][0]['message']['content'].replace('\n', '')
    print(summary)
    response = openai.ChatCompletion.create(
        engine=OPENAI_CHAT_ENGINE,
        messages=[{"role":"system","content":"You are the lead engineer who performs the technology selection for the client's requirements.. Please itemize the names of 30 specific skills. \ne.g.) C, C#, Javascript. \nYou MUST return English.\n"},{"role":"user","content":"We want to create a platform for music that We automatically change the data of music submitted by users to the best data for streaming and distribute it."},{"role":"assistant","content":"Skill needed: API development,SDK development,AI integration,Data analysis,Database management,Web development,Cloud computing,Security protocols"},{"role":"user","content":"Skills should preferably be a reference to a programming language or framework."},{"role":"assistant","content":"Skill needed: Python,Django,Ruby,Ruby on Rails,JavaScript,TypeScript,Express,Node.js,React,Vue,Next.js,Azure,AWS"},{"role":"user","content": input_text},{"role":"assistant","content": summary},{"role":"user","content":"Skills should preferably be a reference to a programming language or framework."}],
        temperature=0.1,
        max_tokens=2045,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    response = response['choices'][0]['message']['content']
    response = response.replace('Skills needed:', '')
    response = response.rstrip('.')
    response = response.split(',')
    for i in range(len(response)):

        if response[i][0] == ' ':
            response[i] = response[i].lstrip()

    return response