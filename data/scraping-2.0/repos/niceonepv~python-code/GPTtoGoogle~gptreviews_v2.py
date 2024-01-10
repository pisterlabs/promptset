import os
import openai
import pandas as pd
import httplib2 
from googleapiclient import discovery
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import gspread
import time
import numpy as np


# Авторизация в OPENAI
openai.organization = "org-bTI6lp1QnSelVMz4isrFbgDx"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Model.list()


# Авторизация в Google
CREDENTIALS_FILE = 'banded-cable-385612-0ca06f32ab93.json'  # Имя файла с закрытым ключом, вы должны подставить свое
scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
credentials = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=scopes)
gc = gspread.authorize(credentials)
gauth = GoogleAuth()
drive = GoogleDrive(gauth)

gs = gc.open_by_url('https://docs.google.com/spreadsheets/d/10xVeqL_8uewOhVvP2SWibbfj7VhAFVjBF-1STVcc_6c/edit#gid=0')
# select a work sheet from its name
worksheet1 = gs.add_worksheet(title='reviews', rows=30, cols=11)

# Читаем запросы из файлов и делаем из них список
def read_texts () -> list:
  reqs = []
  paths = ['./GPTtoGoogle/text_shorts_male.txt', './GPTtoGoogle/text_shorts_female.txt', './GPTtoGoogle/text_medium_male.txt',
           './GPTtoGoogle/text_medium_female.txt', './GPTtoGoogle/text_longlead_male.txt', './GPTtoGoogle/text_longlead_female.txt',
           './GPTtoGoogle/text_poems_male.txt', './GPTtoGoogle/text_poems_female.txt', './GPTtoGoogle/text_army_male.txt', './GPTtoGoogle/text_army_female.txt']
  for path in paths:
      file = open(path, 'r', encoding='utf-8')
      req = file.read()
      reqs.append(req)
      file.close()
  return reqs

def write_to_history(answers:list) -> None:
    history_paths= ['./GPTtoGoogle/history_shorts_male.txt', './GPTtoGoogle/history_shorts_female.txt', './GPTtoGoogle/history_medium_male.txt',
           './GPTtoGoogle/history_medium_female.txt', './GPTtoGoogle/history_longlead_male.txt', './GPTtoGoogle/history_longlead_female.txt',
           './GPTtoGoogle/history_poems_male.txt', './GPTtoGoogle/history_poems_female.txt', './GPTtoGoogle/history_army_male.txt', './GPTtoGoogle/history_army_female.txt']
    for i in range(len(answers)):
        file = open(history_paths[i],'a+', encoding='utf-8')
        file.write('\n'.join(answers[i]))
        file.close()
    return None

def read_from_history() -> list:
    history_paths= ['./GPTtoGoogle/history_shorts_male.txt', './GPTtoGoogle/history_shorts_female.txt', './GPTtoGoogle/history_medium_male.txt',
           './GPTtoGoogle/history_medium_female.txt', './GPTtoGoogle/history_longlead_male.txt', './GPTtoGoogle/history_longlead_female.txt',
           './GPTtoGoogle/history_poems_male.txt', './GPTtoGoogle/history_poems_female.txt', './GPTtoGoogle/history_army_male.txt', './GPTtoGoogle/history_army_female.txt']
    history = []
    for path in history_paths:
        file = open(path, 'r', encoding='utf-8')
        history.append(file.read())
    history = [x.strip() for x in history]
    return history
    

def create_answers(reqs:list) -> list:
  def split_answer(answers:list) -> list:
    splited_answers = []
    lst = []
    for elem in answers:
        lst = elem.split('\n')
        splited_answers.append(lst)
    return splited_answers

  def clear_digits(answers: list) -> list:
    cleared_answers = []
    for answer in answers:
        clear_answer = []
        for string in answer:
            clear_answer.append(string[3:])
        clear_answer = [x.strip() for x in clear_answer]
        cleared_answers.append(clear_answer)
    return cleared_answers

  answers = []
  system_header = './GPTtoGoogle/text_system.txt'
  history = read_from_history()
  for i in range(len(reqs)):
        print(reqs[i])
        result = []
        while len(result) == 0:
             try:
                 completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": system_header},
                                                                                            {"role": "user", "content": reqs[i] + '\n'+ 'Отзыв не должен находиться в списке ниже:'+'\n'+ history[i]}])
                 result = completion.choices[0].message['content']
             except:
                  print('Error')
                  time.sleep(60)
                  continue
        answers.append(result)
  map_func = lambda x: x.replace('\n\n', '\n')
  answers = list(map(map_func, answers))
  splited_answers = split_answer(answers)
  cleared_answers = clear_digits(splited_answers)
  write_to_history(cleared_answers)
  return cleared_answers

def create_table (answers:list) -> pd.DataFrame:
    dictionary = dict(shorts_male = np.array(answers[0]), shorts_female = np.array(answers[1]), medium_male = np.array(answers[2]),
                  medium_female = np.array(answers[3]), longlead_male = np.array(answers[4]), longlead_female = np.array(answers[5]),
                  poems_male = np.array(answers[6]), poems_female = np.array(answers[7]), army_male = np.array(answers[8]),
                  army_female = np.array(answers[9]))
    table = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in dictionary.items()]))
    return table


reqs = read_texts()
answers = create_answers(reqs)
table = create_table(answers)
table.to_csv('./GPTtoGoogle/reviews.csv')

# write to dataframe
worksheet1.clear()
set_with_dataframe(worksheet=worksheet1, dataframe=table, include_index=True,
include_column_header=True, resize=True)          