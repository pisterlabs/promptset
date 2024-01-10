import openai
import os
import pandas as pd
import time
import json
import prompts
import random

# generate your api key on https://platform.openai.com/account/api-keys
# and set up your key in OS (https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety)
openai.api_key = os.environ["OPENAI_API_KEY"]

domains = pd.read_csv("./data/domains.csv")
domains.insert(len(domains.columns), 'company', "")
domains.insert(len(domains.columns), 'company_website', "")
domains.insert(len(domains.columns), 'result', "")

random_indexes = [2694, 3055, 271, 3036, 1741, 1392, 1140, 337, 2819, 1791, 2664, 1354, 1813, 2483, 2176, 2131, 210, 1989, 2002, 2297, 3223, 731, 890, 766, 2189, 2366, 2081, 143, 3535, 2265, 685, 2333, 454, 2006, 439, 1633, 2495, 3216, 2235, 793, 2588, 442, 1548, 3184, 1573, 1574, 2364, 497, 582, 2465, 725, 1740, 2772, 1719, 1116, 3500, 2547, 852, 408, 1234, 2656, 3011, 1060, 2076, 3192, 464, 2111, 1639, 671, 2988, 2265, 807, 3326, 3322, 1410, 1873, 1145, 1668, 2427, 246, 1133, 2256, 1224, 3371, 227, 1536, 3240, 1951, 79, 1872, 1016, 3226, 713, 618, 2981, 2809, 2321, 3513, 780, 372]

temperature = prompts.temperature5
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

for i in random_indexes:
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
    
    

filename = "./data/temperature/prompt6_" + str(temperature).replace(".", "_") + ".csv"

domains.loc[random_indexes].to_csv(filename)  # , index=False




