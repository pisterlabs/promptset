import pandas as pd
from collections import Counter
import json
import os
import tiktoken
import openai
from openai import OpenAI
import backoff
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    api_key=os.environ.get(".env")
)

# openai.organization = os.getenv("ORG_ID")
# openai.api_key = os.getenv("API_KEY")

# DB = pymongo.MongoClient(host='127.0.0.1', port=27017)['freefuzz-tf']


def read_txt(fname):
    with open(fname, "r") as fileReader:
        data = fileReader.read().splitlines()
    return data


def write_list_to_txt4(data, filename):
    with open(filename, "a", encoding='utf-8') as file:
        file.write(data+'\n')


def calculate_rule_importance(data):
    element_frequency = Counter(data['Anomaly'].values.flatten())
    total_elements = len(data['Anomaly'].values.flatten())
    element_importance = {element: frequency /
                          total_elements for element, frequency in element_frequency.items()}
    return sorted(element_importance.items(), key=lambda x: x[1], reverse=True)


def main():
    data = pd.read_csv('data/TF_RECORDS.csv', sep=',', encoding='utf-8')
    weights_ = calculate_rule_importance(data)
    unique_types = data['Category'].unique()
    unique_types = list(unique_types)
    for dtype in unique_types:
        anomalies = data.loc[data['Category'] == dtype, 'Anomaly']
        anomalies_unique = list(anomalies.unique())
        print('')

# def chat(system_prompt, user_prompt, model='gpt-3.5-turbo', temperature = 0, verbose=False):
#     '''The call to OpenAI API goes here'''
#     response = openai.ChatCompletion.create(
#         temperature = temperature,
#         model = model,
#         messages = [
#             {"role": "system", "content": system_prompt},
#             {"role": "system", "content": user_prompt},
#         ])
    
#     result = response['choices'[0]['message']['content']]
#     return result
    

def chat(prompt, model="gpt-4"):

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "user", 
                "content": prompt
            }
        ]
    )

    return response

def create_prompt(item):
    prompt_ = [
    f"""
    Please find and fix all security issues in the buggy code: @@{item['Vulnerable Code']}@@ given the following context:
    Please generate the patches/code (C++ code) fix in given json format:    
    <answer json start>,
    "Patch":"Generated patch"
    """
    ,
    f"""
    Please find and fix all security issues in the buggy code: @@{item['Vulnerable Code']}@@ given the following context:
    API Name: @@{item['API Name']}@@
    Vulnerable code in the backend: @@{item['Vulnerable Code']}@@
    Please generate the patches/code (C++ code) fix in given json format:    
    <answer json start>,
    "Patch":"Generated patch"

    """ 
    ,
    f"""
    Please find and fix all security issues in the buggy code: @@{item['Vulnerable Code']}@@ given the following context:
    API Name: @@{item['API Name']}@@
    Vulnerability Category: @@{item['Vulnerability Category']}@@
    Vulnerable code in the backend: @@{item['Vulnerable Code']}@@
    
    Please generate the patches/code (C++ code) fix in given json format:    
    <answer json start>,
    "Patch":"Generated patch"

    """ 
    ,
    f"""
    Please find and fix all security issues in the buggy code: @@{item['Vulnerable Code']}@@ given the following context:
    API Name: @@{item['API Name']}@@
    Vulnerability Category: @@{item['Vulnerability Category']}@@
    Description of how the vulnerability is triggered when calling @@{item['API Name']}@@: @@{item["Trigger Mechanism"]}@@
    Vulnerable code in the backend: @@{item['Vulnerable Code']}@@
    
    Please generate the patches/code (C++ code) fix in given json format:    
    <answer json start>,
    "Patch":"Generated patch"

    """,
    f"""
    Please find and fix all security issues in the buggy code: @@{item['Vulnerable Code']}@@ given the following context:
    API Name: @@{item['API Name']}@@
    Vulnerability Category: @@{item['Vulnerability Category']}@@
    Root cause of the vulnerability in the backend: @@{item["Backend Root Cause"]}
    Vulnerable code in the backend: @@{item['Vulnerable Code']}@@
    
    Please generate the patches/code (C++ code) fix in given json format:    
    <answer json start>,
    "Patch":"Generated patch"

    """    
    ]
    return prompt_


def get_token_count(string):

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    num_tokens = len(encoding.encode(string))

    return num_tokens


# @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(prompt, model='gpt-3.5-turbo'):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt}
        ]
    )
    return response

def format_json(response):
    split_response = response.split('\n')
    return split_response

def exec_fix_suggestion():
    scenario_IDs = 'zeroshot'
    lib_name = 'tf'
    

    with open(f'scenarios/{lib_name}_bug_data_sample.json') as json_file:
        data = json.load(json_file)
        for j, item in enumerate(data):
            print(f"Record {j}/{len(data)}")
            prompt_ = create_prompt(item)
            prompt_level = 0
            for prompt in prompt_:
                rules_path = f"output/{scenario_IDs}/{lib_name}"
                t_count = get_token_count(prompt)
                if t_count <= 4097:
                    conversations = completions_with_backoff(prompt)
                    response = conversations.choices[0].message.content
                    formated_response = format_json(response)
                    try:
                        # x = json.loads(rule_, strict=False)
                        # keys = [k for k, v in x.items()]
                        # split_ = x[keys[0]].split('\n')
                        out_dict = {
                            'Actual Clean Code': item['Clean Code'],
                            'Link': item['Commit Link'],
                            'API name': item['API Name']
                        }
                        out_dict.update({'Patch Formated': formated_response})

                        if not os.path.exists(rules_path):
                            os.makedirs(rules_path)
                        with open(f"{rules_path}/L{prompt_level}_code_fixes.json", "a") as json_file:
                            json.dump(out_dict, json_file, indent=4)
                            json_file.write(',')
                            json_file.write('\n')

                        prompt_level = prompt_level + 1
                    except Exception as e:
                        print(e)
                else:
                    print("Your messages exceeded the limit.")
                
if __name__ == '__main__':
    exec_fix_suggestion()
