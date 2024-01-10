"""API key = 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
Organization ID org-bTI6lp1QnSelVMz4isrFbgDx
pythonacc@banded-cable-385612.iam.gserviceaccount.comзшз
"""

import os
import openai
import pandas as pd
import httplib2 
from googleapiclient import discovery
from oauth2client.service_account import ServiceAccountCredentials

openai.organization = "org-bTI6lp1QnSelVMz4isrFbgDx"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Model.list()


CREDENTIALS_FILE = 'banded-cable-385612-0ca06f32ab93.json'  # Имя файла с закрытым ключом, вы должны подставить свое
credentials = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive'])
httpAuth = credentials.authorize(httplib2.Http()) # Авторизуемся в системе
# service = discovery.build('sheets', 'v4', http = httpAuth) # Выбираем работу с таблицами и 4 версию API 
# spreadsheet = service.spreadsheets().create(body = {
#     'properties': {'title': 'Первый тестовый документ', 'locale': 'ru_RU'},
#     'sheets': [{'properties': {'sheetType': 'GRID',
#                                'sheetId': 0,
#                                'title': 'Лист номер один',
#                                'gridProperties': {'rowCount': 100, 'columnCount': 15}}}]
# }).execute()
# spreadsheetId = spreadsheet['spreadsheetId'] # сохраняем идентификатор файла
# print('https://docs.google.com/spreadsheets/d/' + spreadsheetId)


# driveService = discovery.build('drive', 'v3', http = httpAuth) # Выбираем работу с Google Drive и 3 версию API
# access = driveService.permissions().create(
#     fileId = '10xVeqL_8uewOhVvP2SWibbfj7VhAFVjBF-1STVcc_6c',
#     body = {'type': 'user', 'role': 'writer', 'emailAddress': 'niceone.pv@gmail.com'},  # Открываем доступ на редактирование
#     fields = 'id'
# ).execute()

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
  return reqs
  
  
  
# def create_reviews(reqs:list) -> list:
#   comp_shorts_male = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#       {"role": "user", "content": reqs[0]}
#     ],
#     n=26
#   )
#   comp_shorts_female = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#       {"role": "user", "content": reqs[1]}
#     ],
#     n=9
#   )  
#   comp_medium_male = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#       {"role": "user", "content": reqs[2]}
#     ],
#     n=21
#   )
#   comp_medium_female = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#       {"role": "user", "content": reqs[3]}
#     ],
#     n=9
#   )
#   comp_longlead_male = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#       {"role": "user", "content": reqs[4]}
#     ],
#     n=19
#   )
#   comp_longlead_female = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#       {"role": "user", "content": reqs[5]}
#     ],
#     n=6
#   )  
#   comp_poems_male = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#       {"role": "user", "content": reqs[6]}
#     ],
#     n=8
#   )
#   comp_poems_female = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#       {"role": "user", "content": reqs[7]}
#     ],
#     n=2
#   )
#   return [comp_shorts_male, comp_shorts_female, comp_medium_male, comp_medium_female, comp_longlead_male, comp_longlead_female, comp_poems_male, comp_poems_female]
# ,
#           comp_shorts_male.choices[0].message['content'], comp_shorts_female.choices[0].message['content'], , comp_medium_female.choices[0].message['content'], comp_longlead_male.choices[0].message['content'],
#           comp_longlead_female.choices[0].message['content'], comp_poems_male.choices[0].message['content'], comp_poems_female.choices[0].message['content']]


def create_answer (reqs:list, type:str, quantity:int, gender:str):
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "user", "content": reqs[6]}
    ],
    n=quantity    
  )
  return completion

reqs = read_texts()
ans = create_answer(reqs, type='poems', quantity=8,gender='male')

print(len(ans['choices']))
for i in range(len(ans['choices'])):
  print(ans.choices[i].message['content'])


map_func = lambda x: x.replace('\n\n', '\n')


def split_answer(answers:list) -> list:
  splited_answers = []
  lst = []
  for elem in answers:
    lst = elem.split('\n')
    splited_answers.append(lst)
  return splited_answers

def clear_digits (answer: list) -> list:
  clear_answer = []
  for elem in answer:
    clear_answer.append(elem[3:])
  return clear_answer

def edit_history(answers:list):
  file = open(r'history_shorts.txt', 'w')
  for string in answers[0]:
    file.write(f'{string}\n')
  for string in answers[1]:
    file.write(f'{string}\n')
  file = open(r'history_medium.txt', 'w')
  for string in answers[2]:
    file.write(f'{string}\n')
  for string in answers[3]:
    file.write(f'{string}\n')
  file = open(r'history_longlead.txt', 'w')
  for string in answers[4]:
    file.write(f'{string}\n')
  for string in answers[5]:
    file.write(f'{string}\n')  
  file = open(r'history_poems.txt', 'w')
  for string in answers[6]:
    file.write(f'{string}\n')
  for string in answers[7]:
    file.write(f'{string}\n')
  return None


# print(reqs)

# answers = []
# answers = create_reviews(reqs)
# print(answers)

# answers = list(map(map_func, answers))

# splited_answers = split_answer(answers)

# cleared_answers = []
# for elem in splited_answers:
#   lst = clear_digits(elem)
#   cleared_answers.append(lst)

# print(splited_answers)
# print(cleared_answers)

# edit_history(cleared_answers)
  


# completion = openai.ChatCompletion.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "user", "content": "Hello!"}
#   ]
# )
# print(completion.choices[0].message)

# def create_reviews (type:str, n:int, gender:str):
#   if type == 'shorts'
#     if gender == 'male':
#       completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Напиши мне " + str(n) + " разных положительных отзывов о купленных черных носках, как будто ты их и купил. Максимальное количество слов - 40, минимальное - 15, в каждом отзыве. Отзывы не нумеруй"}
#         ]
#       )
#     else:
#       completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Напиши мне " + str(n) + " разных положительных отзывов о купленных черных носках, как будто ты их и купил. Максимальное количество слов - 40, минимальное - 15, в каждом отзыве"}
#         ]
#       )
#     return(completion.choices[0].message)
#   if type == 'medium':
    


# def split_answer(answer:str) -> list:
#   lst = []
#   lst = answer.split('\n')
#   return lst

# answer = create_reviews(10, 'male')

# splited_answer = split_answer(answer['content'])


# def clear_digits (answer: list) -> list:
#   clear_answer = []
#   for elem in answer:
#     clear_answer.append(elem[3:])
#   return clear_answer

# splited_answer = clear_digits(splited_answer)
# print(splited_answer)