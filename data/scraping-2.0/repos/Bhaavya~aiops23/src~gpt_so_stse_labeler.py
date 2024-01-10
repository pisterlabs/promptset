import os
import openai
import json
from bs4 import BeautifulSoup
from collections import Counter
from time import sleep

# prompt = "Extract the problem and the resolution, if mentioned, from the following text.\n'''\n{}\n'''\nProblem:"
# prompt = "Extract the problem, if mentioned, from the following text.\n'''\n{}\n'''\nProblem:"
prompt = "Extract the resolution, if mentioned, from the following text.\n'''\n{}\n'''\nResolution:"
def query_gpt(prompt):
  response = openai.Completion.create(
  model="text-davinci-002",
  prompt=prompt,
  temperature=0,
  max_tokens=1000,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop=["'''"],
  logprobs = 5
    )

  print(response)
  return response['choices'][0]

def main():
  with open(posts_path) as f:
    data = f.readlines()
    # posts = json.load(f)
  posts = []
  for d in data:
    posts.append(json.loads(d.strip('\n')))

  cnt = 0
  lens = []
  ops = []
  for p in posts:
    sleep(1)
    try:
    
      full_txt = p['text']
      p['resp'] = query_gpt(prompt.format(full_txt))
      ops.append(p)
      cnt += 1
    except Exception as e:
      print(e)
      pass

    with open('gpt_labels1_design2_res.json','w') as f:
      json.dump(ops,f)


if __name__ == '__main__':
  openai.api_key = API_KEY

  posts_path = '../data/so_stse/ground_truth.jsonl'
  main()
