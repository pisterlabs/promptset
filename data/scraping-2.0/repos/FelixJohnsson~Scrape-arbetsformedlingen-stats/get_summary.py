import openai
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

question = """
Write a new title. (Will be used later to place the jobs in categories).
Write a short summary about this role for a developer. 

What are the requirements? (Short, it is crucial to mention all requirements)
What languages? (It is crucial to mention all languages)
What frameworks? (It is crucial to mention all frameworks)
What education? 
How many years experience is needed? (If there isnt a number in the description, estimate one)
How senior is this role on a scale 1-100. (1 being super junior, 100 being super senior)
What is the email where you could send a resume? (If mentioned)

title: '',
summary: '',
requirements: '', !important
languages: '', !important
frameworks: '', !important
other technologies: '',
education: '', 
years_experience: '',
seniority: '',
email: '',

Reply with English and in a JSON format.
"""

def call_chatGPT(prefix, job):
    text = {"role": "user", "content": prefix + job}

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }
    data = {
        "model": "gpt-4",
        "messages": [text],
        "temperature": 0.7,
    }
    chatGPT_response = requests.post(url, headers=headers, data=json.dumps(data))
    content = chatGPT_response.json()["choices"][0]["message"]["content"]

    print('ChatGPT replied')
    return content

with open('jobs.json', 'r', encoding='utf-8') as f:
    jobs = json.load(f)
    print(len(jobs))

# Loop through each job and get the answer
for job in jobs[30:100]:
    job_str = json.dumps(job)
    answer = call_chatGPT(question, job_str)
    answer_dict = json.loads(answer)

    job.update(answer_dict)

    with open('jobs.json', 'w', encoding='utf-8') as f:
        json.dump(jobs, f, ensure_ascii=False, indent=4)