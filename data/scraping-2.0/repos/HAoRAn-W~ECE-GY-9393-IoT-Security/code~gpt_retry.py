import openai
import os
import pandas as pd
import time
import json
import prompts

openai.api_key = os.environ["OPENAI_API_KEY"]
results = pd.read_csv("./data/full/haoran/prompt1.csv")

total_count = len(results)
temperature = 0.85
prompt = prompts.prompt1

cnt = 0
for i in range(total_count):
    URL = results.iloc[i]["remote_hostname"]
    company = str(results.iloc[i]["company"])
    if company == 'nan':
        print('================' + str(i) + '====================')
        cnt += 1
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
        print(completion.choices[0].message.content)
        try:
            resp = json.loads(completion.choices[0].message.content)
            res = ""
            if 'result' in resp:
                res = resp['result']
            else:
                res = resp['purpose']
            results.loc[i, ['company','company_website','result']] = [resp['company'], resp['company_website'], res]
        except Exception:
            try:
                print('======================= error occured =======================')
                start_index = completion.choices[0].message.content.find('{')
                end_index = completion.choices[0].message.content.rfind('}')
                json_str = completion.choices[0].message.content[start_index:end_index + 1]
                resp = json.loads(json_str)
                res = ""
                if 'result' in resp:
                    res = resp['result']
                else:
                    res = resp['purpose']
                results.loc[i, ['company','company_website','result']] = [resp['company'], resp['company_website'], res]
            except:
                continue
            continue

results.to_csv("./data/full/haoran/prompt1.csv")

print(cnt)