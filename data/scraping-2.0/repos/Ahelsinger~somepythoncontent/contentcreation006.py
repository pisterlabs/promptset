

import openai
import csv
import time

openai.api_key = ""

def generate_rhyming_story(prompt):
    completions = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    message = completions.choices[0].text
    return message

# Prompt for the rhyming story
prompt = "Write a rhyming children's story about a magical unicorn. Make it 200 words, and make each sentence rhyme"

# Generate the rhyming story
rhyming_story = generate_rhyming_story(prompt)

# Split the story into stanzas
stanzas = rhyming_story.split("\n")

# Create a list of field names for the CSV file
fieldnames = []
for i in range(1, len(stanzas)+1):
    fieldnames.append("Verse " + str(i))

# Create a human readable timestamp
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())

# Create a CSV file and write the stanzas as columns
filename = "rhyming_story_" + timestamp + ".csv"
with open(filename, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    row = {}
    for i in range(len(stanzas)):
        row[fieldnames[i]] = stanzas[i]
    writer.writerow(row)
