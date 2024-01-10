import os
import re
from openai import OpenAI
from pathlib import Path

# Specify the directory you want to start from
rootDir = 'tmp'

def replace_score_placeholders (prompt:str, itinerary:str):
  '''
  Replaces the placeholders for city, start_date, end_date, and budget in prompt.

  args:
    prompt: str : prompt used to generate itinerary
    itinerary : str : itinerary

  returns:
    new_prompt : str : the initial prompt with placeholders replaced

  '''
  # Replace the placeholders in the prompt with the random cities, dates, and budget
  new_prompt = re.sub('__itinerary__', itinerary, prompt)
  return new_prompt

def score_itinerary(init_score_prompt, itinerary):
    # Open Client for OpenAI API
    client = OpenAI()
    prompt = replace_score_placeholders(init_score_prompt, itinerary)
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        n=1, # number of chat completion choices to generate for each input message
        temperature=1.0,
        messages=[
        {"role": "user",
        "content": prompt}
        ]
    )
    return completion

with open("utils/ref/score_prompt.txt", "r") as file:
    # Read the entire file as text
    init_score_prompt = file.read()

with open('experiments/itineraries/itinerary_llama2-7b_DPO_Amsterdam_temp-1.0_top-p-1.0_1.txt', 'r') as file:
    # Read the entire file as text
    itinerary = file.read()

# Operate on the text
completion = score_itinerary(init_score_prompt, itinerary)
score = completion.choices[0].message.content

# Output the result as a text file in the same directory
outdir = 'experiments/scores'
with open(os.path.join(outdir, 'itinerary_llama2-7b_DPO_Amsterdam_temp-1.0_top-p-1.0_1' + '_score.txt'), 'w') as file:
    file.write(score)

'''
for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
        # Open the file
        with open(os.path.join(dirName, fname), 'r') as file:
            # Read the entire file as text
            itinerary = file.read()

        # Operate on the text
        completion = score_itinerary(init_score_prompt, itinerary)
        score = completion.choices[0].message.content

        # Output the result as a text file in the same directory
        outdir = os.path.join(dirName, 'scores')
        with open(os.path.join(outdir, Path(fname).stem + '_score.txt'), 'w') as file:
            file.write(score)

    # Prevent os.walk() from walking into subdirectories
    break'''