import time
import csv
import pandas as pd
import openai

openai.organization = "org-ECWHKfp4JLB4quuAdewJ6WuU"
openai.api_key = '<INSERT KEY>'
def prompt_chat_gpt(_prompt):
  response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=_prompt,
    temperature=0.5,
    max_tokens=256,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
  )
  return response['choices'][0]['text']

wikis = pd.read_csv('random-wiki.csv')
title_column = wikis.iloc[:, 1]

titles = wikis.iloc[:, 1].tolist()
wiki_articles = wikis.iloc[:, 0].tolist()
generated_res = []

# init csv
csv_header = ['title', 'generated res', 'wiki article']
with open('generated_responses.csv', 'w', newline='') as f:
  writer = csv.writer(f)
  writer.writerow(csv_header)
  f.close()

for i in range(len(title_column.values)):
  prompt = f"Generate a wikipedia-style article about {titles[i]}. "
  generated_res.append(prompt_chat_gpt(prompt).replace('\n', ''))
  row = [titles[i], generated_res[i], wiki_articles[i].replace('\n', '')]
  with open('generated_responses.csv', 'a', newline='') as f:
      writer = csv.writer(f)
      writer.writerow(row)
      f.close()
  time.sleep(2.0)





