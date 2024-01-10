import os
import openai
import code
import json
openai.api_key = "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"



prompts = json.load(open("prompts.json"))
ids = json.load(open("ids.json"))
tags = []

for idx, prompt in enumerate(prompts[0:500]):

  completion = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,   
  )
  tags.append([ids[idx], completion.choices[0].text])
  print(completion.choices[0].text)
  print(idx)


with open("tags.json", "w") as outfile: 
    json.dump(tags, outfile)


# code.interact(local=dict(globals(), **locals()))

# print(completion.choices[0].text)