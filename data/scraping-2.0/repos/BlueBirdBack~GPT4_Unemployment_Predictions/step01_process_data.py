import csv
import os
import json
import yaml
import openai
from time import time, sleep
from pprint import pprint as pp



###     file operations



def save_yaml(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as file:
        yaml.dump(data, file, allow_unicode=True)



def open_yaml(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data



def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)



def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()



def read_csv_file(file_path):
    with open(file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        data_list = []
        for row in csv_reader:
            if row['O_GROUP'] in ['minor', 'major']:
                data_list.append({
                    'OCC_CODE': row['OCC_CODE'],
                    'OCC_TITLE': row['OCC_TITLE'],
                    'TOT_EMP': row['TOT_EMP']
                })
    return data_list



def chatbot(messages, model="gpt-4-0613", temperature=0):
#def chatbot(messages, model="gpt-3.5-turbo-0613", temperature=0):
    openai.api_key = open_file('key_openai.txt')
    max_retry = 7
    retry = 0
    while True:
        try:
            response = openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature)
            text = response['choices'][0]['message']['content']
            save_yaml('debug/debug_%s.yaml' % time(), messages)
            return text, response['usage']['total_tokens']
        except Exception as oops:
            print(f'\n\nError communicating with OpenAI: "{oops}"')
            if 'maximum context length' in str(oops):
                a = messages.pop(1)
                print('\n\n DEBUG: Trimming oldest message')
                continue
            retry += 1
            if retry >= max_retry:
                print(f"\n\nExiting due to excessive errors in API: {oops}")
                exit(1)
            print(f'\n\nRetrying in {2 ** (retry - 1) * 5} seconds...')
            sleep(2 ** (retry - 1) * 5)



if __name__ == '__main__':
    jobs_data = read_csv_file('BLS_May_2022_All_jobs.csv')
    system_msg = open_file('system.txt')
    for j in jobs_data:
        #### skip if already processed
        
        yaml_file_path = f"jobs/{j['OCC_CODE']} {j['OCC_TITLE']}.yaml"
        if os.path.exists(yaml_file_path):
            print(f"File {yaml_file_path} already exists. Skipping...")
            continue
        
        #### run inference on GPT API
        
        msg = 'Occupation Code: %s\nDescription: %s\nEmployment: %s' % (j['OCC_CODE'], j['OCC_TITLE'], j['TOT_EMP'])
        #print('\n\n\n', j, '\n\n\n', msg)
        employment = int(j['TOT_EMP'].replace(',',''))
        messages = [{'role': 'assistant', 'content': system_msg}, {'role': 'user', 'content': msg}]
        response, tokens = chatbot(messages)
        #print('\n\n\nRESPONSE', response)
        info = json.loads(response)
        merged_info = {**j, **info}
        
        #### calculate jobs lost
        
        lost = employment * info['solo'] * info['vulnerability']
        merged_info['Jobs Lost'] = lost
        merged_info['Jobs Remaining'] = employment - lost
        
        save_yaml(yaml_file_path, merged_info)
        
        print('\n\n\nFINAL\n\n')
        pp(merged_info)
        #exit()