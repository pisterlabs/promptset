import openai
import json
import pandas as pd
from sample_Keywords import sample_cat
import sys
from check_in_cat import check_in_sample
sys.path.append('group_keywors')

API_KEY = open('api_key.txt', 'r').read()
openai.api_key = API_KEY
data = pd.read_csv('14-12-2023-14-05-31 Rawleads_final.csv')
sample_cat = str(sample_cat)




gpt_promt = """below are mapping of categories and job title in python dict format \n {} where the keys are the categories and the values are list of job titles belonging to that category
    I will be giving you some job title and you will tell me which category the job title belongs to in the format [job title] - [category]
""".format(sample_cat)


chat_log = [{"role":"user", "content":gpt_promt}]
initial_response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=chat_log
    )
# print(initial_response)
chat_log.append({"role": "assistant", "content": initial_response.choices[0].message.content})

chat_log.append({"role":"user", "content":"python developer"})
print("Python developer")
response=openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=chat_log
)
assistant_response = response.choices[0].message.content

print(assistant_response)

chat_log.append({"role":"user", "content":"I want your answer to be in the format [job title] belongs to the category [category] and the [category] must be within the keys of the python dictionary provided"})
# if it is not then place it in the most related category
final = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=chat_log
    )

print("FINAL", final.choices[0].message.content)
suggested_cat = []
for index, row in data.iterrows():
    job_title = data['scrape_job_title'].iloc[index]
    exit = check_in_sample(job_title)
    if(exit):
        suggested_cat.append(exit)
    else:
        chat_log.append({"role":"user", "content":job_title})
        response=openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=chat_log
        )
        # assistant_response = response['choices'][0]['message']['content']
        assistant_response = response.choices[0].message.content
        print("answer",index, assistant_response)
        try:
            suggested_cat.append(assistant_response.split('belongs to the category ')[1].replace(".", "").replace('"', "")) if assistant_response.split('belongs to the category')[1] else suggested_cat.append("")
        except:
            suggested_cat.append("")
        chat_log.append({"role":"assistant", "content":assistant_response.strip("\n").strip()})

print("suggested array", suggested_cat)

data["Category"] = suggested_cat

data.to_csv("category_filtering.csv")


