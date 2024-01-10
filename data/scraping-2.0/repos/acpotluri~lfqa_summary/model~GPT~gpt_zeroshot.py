import os
import csv
import json
import random
import openai
from tqdm import tqdm

random.seed(1234)
openai.api_key = os.getenv("OPENAI_API_KEY")

exs = []
with open("test.json", 'r') as f:
    exs = []
    allData = json.load(f)
    for data_id in allData.keys():
        ex = allData[data_id]
        ans = ex['answer_sentences']
        q = ex['question']
        a = ' '.join(ans)
        ns = 0
        for annotation in ['is_summary_1', 'is_summary_2', 'is_summary_3']:
            ns += sum(ex[annotation])
        ns = round(ns / 3) # average number of sentences in the 3 annotated summaries
        print(a, ns)
        exs.append((data_id, q, a, ns))
    f.close()

lengthRestrict = False
out = []
for data_id, q, a, ns in tqdm(exs):
    r = {}
    if lengthRestrict:
        p = f"Q: {q}\n A: {a}\n Summarize the above answer in {ns} sentences.\n"
    else:
        p = f"Q: {q}\n A: {a}\n Summarize the above answer.\n"

    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=p,
        temperature=0,
        top_p=1,
        max_tokens=512,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    r["Data ID"] = data_id
    r["Question"] = q
    r["Answer"] = a
    r["GPT Summary"] = response["choices"][0]["text"][1:]
    out.append(r)

outf = "testset_gpt_limit_noq.csv" if lengthRestrict else "testset_gpt_no_limit_noq.csv"
with open(outf, 'w') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
