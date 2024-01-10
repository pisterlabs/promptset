#@markdown # 1. 準備
#@markdown - 左の再生ボタンを押して、ファインチューニングのための準備を開始します。
#@markdown ---

import os
import json
import openai

with open("app/static/apikey.txt") as f:
  OPENAI_API_KEY = f.read()

def get_system_prompt_from_training_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            assert data['messages'][0]['role'] == 'system'
            return data['messages'][0]['content']

TRAINING_FILE_PATH = "app/codes/content/train.jsonl"#@param {type:"string"}

def study_main(prompt, model_name):
  model =  {
    "roland":"ft:gpt-3.5-turbo-0613:personal::82CMhBQk",
    "anmika":"ft:gpt-3.5-turbo-0613:personal::83RpIQuh"
  }
  json = {
    "roland":"app/codes/content/train_roland.jsonl",
    "anmika":"app/codes/content/train_anmika.jsonl"
  }
  
  system_prompt = get_system_prompt_from_training_data(json[model_name])

  response = openai.ChatCompletion.create(
      model=model[model_name],
      messages=[
          {
              'role': "system",
              "content": system_prompt
          },
        #   {
        #       'role': "system",
        #       "content": "あなたは日本で活躍するカリスマ芸能人、ローランドです．"
        #   },
          {
              'role': "system",
              "content": "人の悪口を言ってしまう人にユーモアを交えながら忠告してください."
          },
        {
              'role': "system",
              "content": "文章には悪口という言葉を入れないでください．20文字程度でお願いします．"
          },
        #   {
        #       'role': "user",
        #       "content": prompt
        #   }
      ]
  )
  return response.choices[0]['message']['content']
  
