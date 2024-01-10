import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
import openai
import tqdm
data = pd.read_excel('./data/小儿哮喘/v1.xlsx')
data.replace(np.nan, '', inplace=True)
data['symptom'] = '性别为:'+data['性别'] + '病诉为:'+data['病诉'] +'病史为:' + data['病史'] + '专科检查结果为:'+data['专科检查'] + '体格检查结果为:'+data['体格检查/望闻切诊']
GPT_MODEL = "gpt-3.5-turbo"
for i in tqdm.tqdm(range(len(data))):
    query = data['symptom'][i]
    time.sleep(25)
    retry_count = 0
    success = False
    
    while not success and retry_count < 3:
        try:
            time.sleep(25)
            response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=[
                {'role': 'system', 'content': '根据下面的文本，识别出其中的病症，并消除相同症状。格式为：症状1、症状2、症状3...。注意仅需要症状，不需要病因、诊断、治疗等。'},
                {'role': 'user', 'content': query},
            ],
            temperature=0,
            )
            success = True
        except:
            retry_count += 1
            time.sleep(25)
            print(f'{i} retrying...')
    if not success:
        print(f"请求失败，已达到最大重试次数：{query}")
        continue
    with open('./output/chatgpt_symptom.txt', 'a', encoding='utf-8') as f:
        f.write(str(i+1) + '\t\t')
        if success:
            f.write(response['choices'][0]['message']['content'].replace('\n',' ') + '\n')