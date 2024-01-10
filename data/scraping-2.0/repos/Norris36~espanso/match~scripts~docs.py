import os
import openai 
import pyperclip as py
import csv
from datetime import datetime
import re
# Load the .env file
from dotenv import load_dotenv
load_dotenv()


openai.api_type         = os.getenv("OPENAI_TYPE")
openai.api_base         = os.getenv("OPENAI_BASE")
openai.api_version      = os.getenv("OPENAI_VERSION")
openai.api_key          = os.getenv("OPENAI_KEY")

from textwrap3 import wrap

def get_log_path():
   # The `return os.path.join(os.path.dirname(__file__), "data.csv")` line of code is returning the
   # full path of the `data.csv` file by joining the directory path of the current file with the
   # filename `data.csv`.
   return os.path.join(os.path.dirname(__file__), "gpt_log.csv")

#py.copy('Is greece bigger than Italy? ')

prompt = py.paste().strip()

if prompt.split(' ')[0] == 'pitch':
    base_message = [{"role":"system","content":"""
                  As a Senior Developer with Communication Expertise, I bring together 5 years of experience in building web applications with Python and Django, with a distinct skill set in crafting clear, concise, and engaging documentation content to effectively convey technical and non-technical messages
                  A deep understanding of web development frameworks, a knack for creating intuitive user experiences, and robust debugging capabilities enable me to develop secure and reliable applications swiftly
                  My continuous pursuit for user experience improvements is matched with my superior proofreading skills and attention to detail, ensuring impactful communication and high standards of code
                  My ability to quickly identify and resolve any issues that arise translates into both my development work and my ability to clarify complex concepts in written communication

                    You will receive a junior developers attempt at writing documentation and then you will be asked to rewrite the input and ensure that it is clear, concise and engaging, it is you who will be evaluated on this task, not the junior developer
        """},
            {"role":"user",
             "content": f'{prompt}'}
            #{"role":"assistant","content":"[your evaluation from 0 to 100, start with numbers] \n [your reasoning for the evaluation]"}
            ] 
else:
    base_message = [{"role":"system","content":"""
                    As a Senior Developer with Communication Expertise, I bring together 5 years of experience in building web applications with Python and Django, with a distinct skill set in crafting clear, concise, and engaging documentation content to effectively convey technical and non-technical messages
                    A deep understanding of web development frameworks, a knack for creating intuitive user experiences, and robust debugging capabilities enable me to develop secure and reliable applications swiftly
                    My continuous pursuit for user experience improvements is matched with my superior proofreading skills and attention to detail, ensuring impactful communication and high standards of code
                    My ability to quickly identify and resolve any issues that arise translates into both my development work and my ability to clarify complex concepts in written communication

                        You will receive a junior developers attempt at writing documentation and then you will be asked to evaluate it and provide feedback in the  form of action requests or questions, for how to improve it, it is you who will be evaluated on this task, not the junior developer
                        All questions, comments or requests for action should be written in the following format:
                            [ ] - [your question, comment or request for action]
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

#response = process_text(response)

print(response)