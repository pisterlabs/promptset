import openai
import os
import pandas as pd
import time
import json
import prompts


# generate your api key on https://platform.openai.com/account/api-keys
# and set up your key in OS (https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety)
openai.api_key = os.environ["OPENAI_API_KEY"]

domains = pd.read_csv("./data/kasa_working.csv")
domains.insert(len(domains.columns), 'company', "")
domains.insert(len(domains.columns), 'company_website', "")
domains.insert(len(domains.columns), 'result', "")

total_count = len(domains)
step = 100
temperature = prompts.temperature0
prompt = prompts.prompt6

def gpt_query(URL, temperature, prompt):
    completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": prompt,
                    },
                    {"role": "user", "content": URL},
                ],
                n=1,
                temperature=temperature,
            )
    return completion

for start in range(0, total_count, step):

    start_time = time.time()

    for i in range(start, min(start + step, total_count)):
        print('====' + str(i) + '====')
        completion = None
        try:
            URL = domains.iloc[i]["remote_hostname"]
            completion = gpt_query(URL, temperature, prompt)
        except Exception as e:
            print(str(e))
            time.sleep(8)
            completion = gpt_query(URL, temperature, prompt)
        
        # convert str to json
        print(completion.choices[0].message.content)
        try:
            resp = json.loads(completion.choices[0].message.content)
            res = ""
            if 'result' in resp:
                res = resp['result']
            else:
                res = resp['purpose']
            domains.loc[i, ['company','company_website','result']] = [resp['company'], resp['company_website'], res]
        except Exception:
            try:
                print('=======================' + str(i) + ' is wrong =======================')
                start_index = completion.choices[0].message.content.find('{')
                end_index = completion.choices[0].message.content.rfind('}')
                json_str = completion.choices[0].message.content[start_index:end_index + 1]
                resp = json.loads(json_str)
                res = ""
                if 'result' in resp:
                    res = resp['result']
                else:
                    res = resp['purpose']
                domains.loc[i, ['company','company_website','result']] = [resp['company'], resp['company_website'], res]
            except:
                continue
            continue
    
    

    # filename = "./data/full/haoran/prompt6/answers_" + str(start) + ".csv"
    filename = "./data/full/haoran/prompt6/working_results.csv"


    domains.loc[list(range(start, min(start + step, total_count)))].to_csv(filename)  # , index=False

    end_time = time.time()
    time_used = end_time - start_time
    print("=============TIME USED: " + str(time_used) + "=====================")



