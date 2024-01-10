import openai
import argparse
import pandas as pd
import os 
import base64
import requests
import time

# create user inputs
def get_user_input():
    parser = argparse.ArgumentParser(description='ONET API User Input')
    parser.add_argument('-u', help = "ONET API Username")
    parser.add_argument('-p', help = "ONET API Password")
    parser.add_argument('-f', help = "PATH to file with survey responses - should be first column of it")
    parser.add_argument('-k', help = "openai key")
    parser.add_argument("-r", help = "either 0 or the number of rows to randomly sample from the survey responses")

    args = parser.parse_args()

    return args.username, args.password, args.path, args.key, args.random



## LLM FUNCTIONS
def extract_occupation(survey_response):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful research assistant. Your task is to extract the occupation(s) from the provided survey response. If you are uncertain, make a guess."
            },
            {
                "role": "user",
                "content": survey_response
            }
        ]
    )

    return response['choices'][0]['message']['content']

def estimate_soc_code(occupation):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful research assistant. Your task is to estimate the major-group SOC code as per the BLS definition for the provided occupation. If you are uncertain, make a guess."
            },
            {
                "role": "user",
                "content": occupation
            }
        ]
    )

    return response['choices'][0]['message']['content']

# for choosing the SOC group
def choose_soc_group(job_title, soc_groups):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful research assistant. Your task is to select the most likely Standard Occupational Classification (SOC) group for the provided job title from the given list. If you are uncertain, make a guess."
            },
            {
                "role": "user",
                "content": f"Job title: {job_title}\nPotential SOC groups: {', '.join(soc_groups)}"
            }
        ]
    )

    return response['choices'][0]['message']['content']

## O*NET API FUNCTIONS
class OnetWebService:
    def __init__(self, username, password):
        self._headers = {
            'User-Agent': 'python-OnetWebService/1.00 (bot)',
            'Authorization': 'Basic ' + base64.b64encode((username + ':' + password).encode()).decode(),
            'Accept': 'application/json'
        }
        self._url_root = 'https://services.onetcenter.org/ws/'

    def call(self, path, query=None):
        url = self._url_root + path
        response = requests.get(url, headers=self._headers, params=query)
        if response.status_code == 200:
            return response.json()
        else:
            return {
                'error': 'Call to ' + url + ' failed with error code ' + str(response.status_code),
                'response': response.text
            }
        
def get_soc_info(occupation_title, username, password):
    onet_ws = OnetWebService(username, password)
    soc_info = onet_ws.call('online/search', {'keyword': occupation_title, "end" : 5})

    rel_score = []
    code = []
    title = []
    for content in soc_info['occupation']:
        rel_score.append(content['relevance_score'])
        code.append(content['code'])
        title.append(content['title'])

    soc_info = pd.DataFrame({'relevance_score': rel_score, 'code': code, 'title': title})
    soc_info["major_group"] = soc_info["code"].str[:2]

    return soc_info


## create main function

def main():
    # 0. get user input
    username, password, path, api_key, random = get_user_input()

    # 1. import job titles
    df = pd.read_csv(path + "/" "job_titles.csv")

    # if we just want a subset of the data
    if random != "0":
        df = df.sample(n = int(random))
    

    # api key
    openai.api_key = api_key

    # 3. init empty lists
    occupations = []
    soc_codes = []
    soc_info_dfs = []

    master_df = pd.DataFrame()

    for i, surv_response in enumerate(df[:, 0]):
        print(f"processing survey response {i} of {len(df)}")

        extracted_occupation = extract_occupation(surv_response)
        occupations.append(extracted_occupation)

        # estimate SOC code
        estimated_soc_code = estimate_soc_code(extracted_occupation)
        soc_codes.append(estimated_soc_code)

        soc_info_df = get_soc_info(extract_occupation, username, password)
        soc_info_dfs.append(soc_info_df)

        # choose SOC group
        titles = soc_info_df["title"]
        optimal_title = choose_soc_group(extracted_occupation, titles)
        index = titles.index(optimal_title)

        df_row = [extract_occupation, estimate_soc_code, soc_info_df.iloc[index, :]]

        master_df.append(df_row)
        time.sleep(0.5)

    
    master_df.to_csv(path + "/" + "master_occupations.csv")
    
    return 0




if __name__ == "__main__":
    main()


        







