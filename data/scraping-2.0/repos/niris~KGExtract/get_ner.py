import os
import sys
import openai
import re
import csv

openai.api_key = os.getenv("OPENAI_API_KEY")

with open(os.path.join("input",sys.argv[1]), 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    f = open(os.path.join("output","ner_dataset.csv"), "a")
    for row in csvreader:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt="Without any description, generate ner tagging dataset in BOI format with a list of  tokens and a list of labels like in python list format for example tokens=['word1','word2','word3'],labels=['O','B-PERSON','I-PERSON'],  for the following text:" 
                    + row[1],
            temperature=0.3,
            max_tokens=1000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        tokens = re.findall(r'\[.*?\]', response["choices"][0]["text"])
        f.write("\n")
        f.write(f'{csvreader.line_num}, {row[1]},')
        for token in tokens:
            f.write(f'{token}, ')
    f.close()
