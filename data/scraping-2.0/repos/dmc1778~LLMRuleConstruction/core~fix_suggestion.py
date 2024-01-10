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


def gpt_conversation(prompt, model="gpt-4"):

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


def create_prompt_fix_sugesstion(item):
    prompt_ = f"""
    You are a chatbot responsible for suggesting patches that fixing bugs in deep learning pipeline codes written in Python.
    For each bug, you have the following information:
    Bug description: {item['Bug description']}
    Bug triggering code: {item['Sample Code']}
    
    Please generate the patches in given json format:
        
    <answer json start>,
    "Patch":"Generated patch"

    """

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


def exec_fix_suggestion():
    lib_name = 'tf'
    rules_path = f"output/{lib_name}_patches_new.json"

    with open(f'data/{lib_name}_bug_data.json') as json_file:
        data = json.load(json_file)
        for j, item in enumerate(data):
            print(f"Record {j}/{len(data)}")
            prompt_ = create_prompt_fix_sugesstion(item)
            t_count = get_token_count(prompt_)
            if t_count <= 4097:
                conversations = completions_with_backoff(prompt_)
                rule_ = conversations.choices[0].message.content
                
                try:
                    x = json.loads(rule_)
                    x.update({'Link': item['Link']})
                    x.update({'API name': item['API Signature']})
                    x.update({'Bug description': item['Bug description'] })
                    with open(rules_path, "a") as json_file:
                        json.dump(x, json_file, indent=4)
                        json_file.write(',')
                        json_file.write('\n')
                except Exception as e:
                    print(e)
            else:
                print("Your messages exceeded the limit.")
                
if __name__ == '__main__':
    exec_fix_suggestion()
