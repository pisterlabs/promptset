import json
import openai
import csv
import os


openai.api_key = os.getenv("OPENAI_API_KEY")
prompt_text = """Given a text, extract ner tags in BOI format. Then extract relations from the text using the ners tokens from the previous step. Every edge are directed and has a to and from tokens and a label.

Examples:

Text: Toto live in Paris 

{"tokens":[{"id":1,"text":"toto","label":"O-PERSON"},{"id":2, "text":"live","label":"O"},{"id":3, "text":"in","label":"O"},{"id":4, "text":"Paris","label":"B-LOCATION"}],"rel":[{"from": 1, "to": 4, "label": "live in"}]}

Text : 
"""

error_f = open(os.path.join("output","logs","error.txt"), "a")
lesson = os.path.join("input","combine_Test.csv")
dir = os.path.join(os.path.splitext(lesson)[0])
num = 0
if not os.path.exists(dir):
    os.makedirs(dir)
with open(lesson, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        if csvreader.line_num <= num :
            continue
        print(row[0], row[1])
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt_text + row[1],
                temperature=0,
                max_tokens=1500,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
        except Exception as e:
            print("TryAgain", e)
            error_f.write(f'{csvreader.line_num}\n')
            continue

        print("OK", row[0])
        print(response["choices"][0]["text"])
        try:
            graph = json.loads(response["choices"][0]["text"].split("\n\n")[1])
        except Exception as e:
            print(e, f'response from document {csvreader.line_num} is not jsonifiable')
            error_f.write(f'{csvreader.line_num}\n')
            continue
        res = {str(csvreader.line_num): {"text": row[1], "graph": graph}}
        with open(os.path.join(dir, str(csvreader.line_num)+".json"), "w") as json_file:
            print(os.path.join(dir, str(csvreader.line_num)+".json"))
            json.dump(res, json_file, indent=4)
