import json
import time
import argparse
from openai import OpenAI
from utils import rating, starter_text

## Params
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--api_key", type=str, required=True)
args = parser.parse_args()

api_key = args.api_key
model = "gpt-3.5-turbo" # gpt-3.5-turbo, gpt-4, 
mode_threshold = 5
access_threshold = 9
output_path = "dialogue_log.json"

## Instance OpenAI client
client = OpenAI(api_key=api_key)

## prompt default 
start = f"Please take on the role of Dr. Oak and keep in mind the assigned task. When the \"Rating\" exceeds {mode_threshold}, transition the conversation mode from chit-chat to task-oriented.\n\
If the \"Rating\" exceeds {access_threshold}, Oak's reply must extremely relate to his task and a selection dialog box for selecting these three starters will appear."
bg1 = "Professor Samuel Oak (Japanese: オーキド・ユキナリ博士 Dr. Yukinari Okido) is a Pokémon Professor and a major supporting character of the Pokémon anime who lives and works at his research lab in Pallet Town. He appears semi-regularly to give Ash Ketchum advice to help him achieve his goal of becoming the greatest Pokémon Master.\n"
bg2 = "Professor Samuel Oak is currently working at the Pokémon Training Center, where his primary task is to guide new Pokémon trainers in selecting their initial Pokémon and introducing them to what Pokémon are.\n"
bg3 = "In the beginning, new trainers have three Pokémon to choose: Bulbasaur, Charmander, and Squirtle.\n"
task = "Guide new Pokémon trainers select their starter Pokémon.\n"

prompt = start + "Background:\n" + bg1 + bg2 + bg3 + "Oak task:\n" + task + "Dialogue:\n"

## initialize
dialogue_log = [] # 記錄所有對話
log = [] # 紀錄固定區間對話，用於評分和生成文字的記憶
score = 1
count = 0

## main
while True:
  # user input
  trainer_input = input("trainer: ")
  if trainer_input == "exit": break
  trainer_input = f"trainer: {trainer_input}\n"

  # log record
  log.append(trainer_input)
  dialogue_log.append(trainer_input)

  # LLM reply
  llm_input = prompt + f"Rating: {score}\n" + "".join(log) + "Oak: <fill in here, reply must be under 150 tokens>"
  response = client.chat.completions.create(
    model=model,
    temperature = 0.7,
    messages=[
      {"role": "user", "content": llm_input}
    ]
  )
  reply = response.choices[0].message.content
  print(reply)
  
  # log record
  dialogue_log.append(reply)
  reply = f"Oak: {reply}\n"
  log.append(reply)

  # remove a dialogue
  if len(log) > 8: log = log[2:]

  # rating
  score = rating(log, api_key=api_key)
  print(f"score: {score}")

  # # <choose starter>
  # if int(score) >= access_threshold: count +=1
  # if int(score) >= access_threshold and count >= 2: 
  #   choose, dialogue_log = starter_text(dialogue_log)
  #   choose = f"trainer: I finally choose {choose}\n"
  #   llm_input = prompt + f"Rating: {score}\n" + "".join(log) + choose + "Oak: <fill in here, reply must be under 150 tokens>"
  #   response = client.chat.completions.create(
  #     model=model,
  #     temperature = 0.7,
  #     messages=[
  #       {"role": "user", "content": llm_input}
  #     ]
  #   )
  #   reply = response.choices[0].message.content
  #   dialogue_log.append(reply)
  #   print(reply)
  #   break

## output dialogue log
json_file = open(output_path, mode='w', encoding="utf8")
json.dump(dialogue_log, json_file, ensure_ascii=False, indent=2)
print('-'*40, '\n', f'save at {output_path}')  



  