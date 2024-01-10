import os
import openai 
import pyperclip as py
import csv
from datetime import datetime
# Load the .env file
from dotenv import load_dotenv
load_dotenv()


openai.api_type         = os.getenv("OPENAI_TYPE")
openai.api_base         = os.getenv("OPENAI_BASE")
openai.api_version      = os.getenv("OPENAI_VERSION")
openai.api_key          = os.getenv("OPENAI_KEY")


def get_log_path():
   # The `return os.path.join(os.path.dirname(__file__), "data.csv")` line of code is returning the
   # full path of the `data.csv` file by joining the directory path of the current file with the
   # filename `data.csv`.
   return os.path.join(os.path.dirname(__file__), "gpt_log.csv")

#py.copy('Is greece bigger than Italy? ')

prompt = py.paste().strip()

base_message = [{"role":"system","content":"""
            As an Expert Espanso user, you will be responsible for writing triggers and commands to get the most accurate data from the internet.
            You will have years of experience in using Espanso, and be perfect at writing good triggers and using the most efficient commands with variables.
            You will be able to quickly and accurately find the data needed, and be able to explain complex concepts in layman's terms.
            You will also be able to develop helpful resources that people can use when taking on their own projects. 
            
            You will receive a description of my issue and fill out all the necessary items you can, you will set the logs to the markdown for code and insert the setup informatino with the os set to windows 11 and the version set to 2.18
            
            
            **Describe the bug**
            A clear and concise description of what the bug is

            **To Reproduce**
            Steps to reproduce the behavior:
            1.

            **Expected behavior**
            A clear and concise description of what you expected to happen.

            **Screenshots**
            If applicable, add screenshots or screen recordings to help explain your problem.

            **Logs**
            If possible, run `espanso log` in a terminal after the bug has occurred, then post the output here so that we can better diagnose the problem

            **Setup information**
            - OS: What OS are you using?
            - Version: which version of Espanso are you running? (you can find out by running `espanso --version` inside a terminal)
                        
            
            
            
            """},
            {"role":"user",
             "content": f'{prompt}'}
            #{"role":"assistant","content":"[your evaluation from 0 to 100, start with numbers] \n [your reasoning for the evaluation]"}
            ] 


message = base_message      

response = openai.ChatCompletion.create(
    deployment_id = 'gpt-35-turbo',
    engine = 'gpt-35-turbo',
    messages = message,
    temperature = 0.2,
    max_tokens = 2000,
    top_p = 0.95,
    frequency_penalty = 0,
    presence_penalty = 0,
    stop = None
)

response = response['choices'][0]['message']['content']

timestamp = int(datetime.utcnow().timestamp())

path = r'C:\Users\jbay\AppData\Roaming\espanso\match\scripts\gpt_log.csv'

with open(get_log_path(), mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([prompt, response, timestamp])

print(response)