import openai
import pandas as pd
from collections import ChainMap
# Set your OpenAI API key here
api_key = "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"

data = pd.read_csv('linkedinjobs5.csv')

# Convert the DataFrame into a list of dictionaries
jobs = data.to_dict(orient='records')
print(jobs)

def extract_skills_from_description(job_description):
    prompt = f"Extract only the top 5 technical skills etc from the following job description and return result in the form of comma separated values: '{job_description}'"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=1.0,
        api_key=api_key,
    )
    extracted_skills = response.choices[0].text.strip()
    return extracted_skills
final_dict2=[]
# Loop over your job descriptions and call the extract_skills_from_description function
for job in jobs:
    extracted_skills_text = extract_skills_from_description(job['description'])
    extracted_skills_list = extracted_skills_text.split(' \n')
    skills_list = []
    for i, skill in enumerate(extracted_skills_list, start=1):
        skills_list.append(skill)
    print(skills_list)
    final_dict = {'company': job['company'], 'job_title': job['job-title'], 'level': job['level'],
                  'skills': skills_list}
    print(final_dict)
    final_dict2.append(final_dict)


df = pd.DataFrame(final_dict2)
df.to_csv('jobswithskills.csv', index=False, encoding='utf-8')
