import os
import openai
import pandas as pd
import httplib2 
from googleapiclient import discovery
from oauth2client.service_account import ServiceAccountCredentials
import time
import pandas as pd
import numpy as np



# Авторизация в OPENAI
openai.organization = "org-bTI6lp1QnSelVMz4isrFbgDx"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Model.list()


# Авторизация в Google
CREDENTIALS_FILE = 'banded-cable-385612-0ca06f32ab93.json'  # Имя файла с закрытым ключом, вы должны подставить свое
credentials = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive'])
httpAuth = credentials.authorize(httplib2.Http()) # Авторизуемся в системе


# Добавление ответов в файл истории
def add_to_history(answers:list):
  file = open(r'./GPTtoGoogle/history_shorts.txt', 'w')
  for string in answers[0]:
    file.write(f'{string}\n')
  for string in answers[1]:
    file.write(f'{string}\n')
  file = open(r'./GPTtoGoogle/history_medium.txt', 'w')
  for string in answers[2]:
    file.write(f'{string}\n')
  for string in answers[3]:
    file.write(f'{string}\n')
  file = open(r'./GPTtoGoogle/history_longlead.txt', 'w')
  for string in answers[4]:
    file.write(f'{string}\n')
  for string in answers[5]:
    file.write(f'{string}\n')  
  file = open(r'./GPTtoGoogle/history_poems.txt', 'w')
  for string in answers[6]:
    file.write(f'{string}\n')
  for string in answers[7]:
    file.write(f'{string}\n')
  return None

# Получаем ответы на наши запросы
def create_answer (req:str, quantity:int, historyfile='./GPTtoGoogle/history.txt'):
  result = []
  file = open(historyfile, 'a+', encoding='utf-8')
  history = file.readlines()
  history = [x.strip() for x in history]
  while len(result) <= quantity:
    try:
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": req}])
        if (completion not in result) and (completion not in history):
            result.append(completion.choices[0].message['content'])
    except:
        print('Error')
        continue
  file.write('\n'.join(result))
  file.close()
  return result
    
    
        
# Читаем запросы из файлов и делаем из них список
def read_texts () -> list:
  reqs = []
  file = open('./GPTtoGoogle/text_shorts_male.txt', 'r', encoding='utf-8')
  req_shorts_male = file.read()
  reqs.append(req_shorts_male)
  file = open('./GPTtoGoogle/text_shorts_female.txt', 'r', encoding='utf-8')
  req_shorts_female = file.read()
  reqs.append(req_shorts_female)
  file = open('./GPTtoGoogle/text_medium_male.txt', 'r', encoding='utf-8')
  req_medium_male = file.read()
  reqs.append(req_medium_male)
  file = open('./GPTtoGoogle/text_medium_female.txt', 'r', encoding='utf-8')
  req_medium_female = file.read()
  reqs.append(req_medium_female)      
  file = open('./GPTtoGoogle/text_longlead_male.txt', 'r', encoding='utf-8')
  req_longlead_male = file.read()
  reqs.append(req_longlead_male)
  file = open('./GPTtoGoogle/text_longlead_female.txt', 'r', encoding='utf-8')
  req_longlead_female = file.read()
  reqs.append(req_longlead_female)  
  file = open('./GPTtoGoogle/text_poems_male.txt', 'r', encoding='utf-8')
  req_poems_male = file.read()
  reqs.append(req_poems_male)
  file = open('./GPTtoGoogle/text_poems_female.txt', 'r', encoding='utf-8')
  req_poems_female = file.read()
  reqs.append(req_poems_female)
  file = open('./GPTtoGoogle/text_army_male.txt', 'r', encoding='utf-8')
  req_army_male = file.read()
  reqs.append(req_army_male)
  file = open('./GPTtoGoogle/text_army_female.txt', 'r', encoding='utf-8')
  req_army_female = file.read()
  reqs.append(req_army_female) 
  file.close() 
  return reqs


def add_answers_to_list (ans_req) -> list:
    result = []
    for i in range(len(ans_req['choices'])):
        result.append(ans_req.choices[i].message['content'])
    return result

    

reqs = read_texts()


ans_shorts_male = create_answer(reqs[0], 26)
ans_shorts_female = create_answer(reqs[1], 6)
ans_medium_male = create_answer(reqs[2], 21)
ans_medium_female = create_answer(reqs[3], 9)
ans_longlead_male = create_answer(reqs[4], 19)
ans_longlead_female = create_answer(reqs[5], 6)
ans_poems_male = create_answer(reqs[6], 8)
ans_poems_female = create_answer(reqs[7], 2)
ans_army_male = create_answer(reqs[8], 4)
ans_army_female = create_answer(reqs[9], 1)

dictionary = dict(shorts_male = np.array(ans_shorts_male), shorts_female = np.array(ans_shorts_female), medium_male = np.array(ans_medium_male),
                  medium_female = np.array(ans_medium_female), longlead_male = np.array(ans_longlead_male), longlead_female = np.array(ans_longlead_female),
                  poems_male = np.array(ans_poems_male), poems_female = np.array(ans_poems_female), army_male = np.array(ans_army_male),
                  army_female = np.array(ans_army_female))

table = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in dictionary.items()]))

print(table)

table.to_csv('./GPTtoGoogle/reviews.csv')