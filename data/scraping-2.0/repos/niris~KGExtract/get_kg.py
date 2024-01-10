import json
import openai
import csv
import os


openai.api_key = os.getenv("OPENAI_API_KEY")
prompt_text = """Given a text, extract a knowledge graph from the text by extrapolating as many relationships as possible from the text. It's important to extract every singles words frow text and store them all in the nodes list. Every node has an id, label, and its NER tag in BOI format. Every edge has a to and from with node ids, and a label. Edges are directed, so the order of the from and to is important.

Examples:

Text: Toto is tata's friend

{ "nodes": [ { "id": 1, "label": "Toto", "ner_tag": "B-PERSON" },{ "id": 2, "label": "is", "ner_tag": "O" }, { "id": 3, "label": "Tata", "ner_tag": "B-PERSON" },{ "id": 4, "label": "friend", "ner_tag": "O" } ], "edges": [ { "from": 1, "to": 3, "label": "friend" } ] }

Text : 
"""

error_f = open(os.path.join("output","logs","error.txt"), "a")
lesson = os.path.join("input","test.csv")
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
                model="text-ada-001",
                prompt=prompt_text + row[1],
                temperature=0,
                max_tokens=1000,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
        except Exception as e:
            print("TryAgain", e)
            error_f.write(f'{csvreader.line_num}\n')
            continue

        print(response)
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
