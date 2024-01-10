import os
from nltk import tokenize
import csv

# import openai

# openai.api_key = os.getenv("OPENAI_API_KEY")

#response = openai.Completion.create(
#  engine="text-davinci-002",
#  prompt="",
#  temperature=1,
#  max_tokens=256,
#  top_p=1,
#  frequency_penalty=2,
#  presence_penalty=2
#)

a_file = open("ceresonepage")
file_contents = a_file.read()
contents_split = file_contents.splitlines()

sentences = []

# tokenize.sent_tokenize(p)

for p in contents_split:
  sentences += tokenize.sent_tokenize(p)

pairs = []

for i in range(len(sentences)):
  if i+1 >= len(sentences): 
    break
  pairs.append({
    "prompt": sentences[i],
    "completion": sentences[i+1]
    })

with open('ceresonepage.csv', 'w',) as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['prompt', 'completion'])
    for pair in pairs:
      writer.writerow([pair["prompt"], pair["completion"]])